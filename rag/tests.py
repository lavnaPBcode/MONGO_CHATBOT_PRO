"""
Run with:  python manage.py test
Needs no real MongoDB, internet or API key: MongoDB is mongomock, embeddings are a
deterministic hashing embedder, and the LLM is faked.
"""
import json
import re
import tempfile
import zlib
from unittest.mock import patch

import mongomock
import numpy as np
from bson import ObjectId
from django.test import Client, SimpleTestCase, override_settings

from rag.management.commands.seed_sample_data import SAMPLE
from rag.services import (document_converter, embedding_service, embedding_store, ingestion_service,
                          llm_service, rag_service, relationships, retrieval, vector_index)
from rag.services.exceptions import InvalidRequestError, VectorStoreError
from rag.services.retrieval import rrf_fuse


def fake_embed_texts(texts, batch_size=64):
    """Bag-of-words hashing embedder: texts sharing words get higher cosine similarity."""
    out = np.zeros((len(texts), 128), dtype="float32")
    for i, text in enumerate(texts):
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            out[i, zlib.crc32(token.encode()) % 128] += 1
        out[i] /= np.linalg.norm(out[i]) or 1
    return out


def _mongomock_add_replace_shim():
    """Newer pymongo passes `sort=` to the bulk builder; older mongomock doesn't accept it (test-only gap)."""
    from mongomock.collection import BulkOperationBuilder

    original = BulkOperationBuilder.add_replace

    def add_replace(self, selector, replacement, upsert=False, collation=None, hint=None, sort=None):
        return original(self, selector, replacement, upsert=upsert, collation=collation, hint=hint)

    return patch.object(BulkOperationBuilder, "add_replace", add_replace)


class MongoCase(SimpleTestCase):
    """Base: fresh in-memory MongoDB with the sample data, fake embedder, temp index dir."""

    def setUp(self):
        self.db = mongomock.MongoClient()["testdb"]
        for name, docs in SAMPLE.items():
            self.db[name].insert_many([dict(d) for d in docs])
        tmp = tempfile.mkdtemp()
        patches = [
            patch("rag.services.mongo_service.get_db", return_value=self.db),
            patch("rag.services.mongo_service.get_embeddings_collection", return_value=self.db["rag_embeddings"]),
            patch("rag.services.mongo_service.ping", return_value=True),
            patch("rag.services.embedding_service.embed_texts", side_effect=fake_embed_texts),
            _mongomock_add_replace_shim(),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        overrides = override_settings(VECTOR_STORE_DIR=tmp, MIN_SCORE=0.0, EMBEDDING_PROVIDER="local")
        overrides.enable()
        self.addCleanup(overrides.disable)
        vector_index.reset_for_tests()
        embedding_service.clear_query_cache()
        rag_service._answer_cache.clear()

    def ingest(self, **kwargs):
        return ingestion_service.run_ingestion(log=lambda m: None, **kwargs)

    def emb(self, source, doc_filter=None):
        return self.db["rag_embeddings"].find_one({"_id": {"$regex": f"^{source}:"}, **(doc_filter or {})})


class ConverterTests(SimpleTestCase):
    def test_types_are_converted_to_readable_text(self):
        from datetime import datetime
        oid = ObjectId()
        doc = {"_id": ObjectId(), "name": "Ann", "tags": ["a", "b"], "active": True,
               "joined": datetime(2026, 8, 12, 6, 38), "owner": oid,
               "address": {"city": "Paris", "geo": {"lat": 1.5}},
               "password": "secret", "empty": "", "blob": b"x", "items": [{"sku": "A1"}, {"sku": "B2"}]}
        text = "\n".join(document_converter.convert_document("people", doc).lines)
        for expected in ("Name: Ann", "Tags: a, b", "Active: Yes", "Joined: August 12, 2026 06:38 UTC",
                         f"Owner ID: {oid}", "Address City: Paris", "Address Geo Lat: 1.5",
                         "Items 1 Sku: A1", "Items 2 Sku: B2"):
            self.assertIn(expected, text)
        for missing in ("secret", "Password", "Empty", "Blob"):
            self.assertNotIn(missing, text)

    def test_metadata_has_primary_id_collection_fields_and_references(self):
        oid, doc_id = ObjectId(), ObjectId()
        record = document_converter.to_records("things", {"_id": doc_id, "label": "X", "ref": oid})[0]
        meta = record["metadata"]
        self.assertEqual(record["_id"], f"things:{doc_id}:0")
        self.assertEqual((meta["collection"], meta["primary_id"]), ("things", str(doc_id)))
        self.assertIn("label", meta["field_names"])
        self.assertEqual(meta["reference_ids"], [str(oid)])

    def test_long_text_is_chunked_and_nothing_is_lost(self):
        words = " ".join(f"w{i}" for i in range(1500)) + " THE_END"
        records = document_converter.to_records("docs", {"_id": "d1", "name": "Report", "body": words})
        self.assertGreater(len(records), 5)
        self.assertTrue(all(len(r["text"]) < 1000 for r in records))
        self.assertIn("THE_END", records[-1]["text"])
        self.assertIn("(Report)", records[1]["text"])
        self.assertEqual(len({r["_id"] for r in records}), len(records))

    def test_empty_document_gives_no_records(self):
        self.assertEqual(document_converter.to_records("c", {"_id": ObjectId()}), [])


class RelationshipTests(MongoCase):
    def test_guess_collections(self):
        names = ["users", "workspaces", "categories", "addresses"]
        g = relationships.guess_collections
        self.assertEqual(g("userId", names), ["users"])
        self.assertEqual(g("workspace", names), ["workspaces"])
        self.assertEqual(g("category_id", names), ["categories"])
        self.assertEqual(g("address", names), ["addresses"])
        self.assertEqual(g("createdBy", names), [])

    def test_detects_targets_by_name_and_by_probing(self):
        resolver = relationships.RelationshipResolver(self.db, list(SAMPLE))
        resolver.prefetch("memberships", list(self.db["memberships"].find()))
        resolver.prefetch("projects", list(self.db["projects"].find()))
        resolver.prefetch("tasks", list(self.db["tasks"].find()))
        t = resolver._targets
        self.assertEqual(t[("memberships", "user")], "users")
        self.assertEqual(t[("memberships", "workspace")], "workspaces")
        self.assertEqual(t[("projects", "workspaceId")], "workspaces")
        self.assertEqual(t[("projects", "createdBy")], "users")       # found by probing
        self.assertEqual(t[("tasks", "assignedTo")], "users")         # found by probing
        self.assertEqual(t[("tasks", "projectId")], "projects")

    def test_string_ids_are_treated_as_references_when_they_resolve(self):
        self.db["notes"].insert_one({"text": "hello", "author": str(ObjectId("b00000000000000000000001"))})
        resolver = relationships.RelationshipResolver(self.db, list(SAMPLE) + ["notes"])
        doc = self.db["notes"].find_one()
        resolver.prefetch("notes", [doc])
        text = "\n".join(document_converter.convert_document("notes", doc, resolver).lines)
        self.assertIn("Author: Aisha Khan", text)


class IngestionTests(MongoCase):
    def test_ingest_stores_text_metadata_relationships_and_vector_together(self):
        summary = self.ingest()
        self.assertEqual(summary["mode"], "incremental")
        doc = self.emb("memberships")
        self.assertEqual(len(doc["embedding"]), 128)
        self.assertIn("User: Aisha Khan", self.db["rag_embeddings"].find_one(
            {"_id": f"memberships:{self.db['memberships'].find_one()['_id']}:0"})["text"])
        self.assertIn("Workspace: Acme Corp", doc["text"])
        self.assertEqual({r["ref_collection"] for r in doc["relationships"]}, {"users", "workspaces"})
        self.assertEqual(doc["metadata"]["model"], "local:sentence-transformers/all-MiniLM-L6-v2")
        for rel in doc["relationships"]:
            self.assertEqual(set(rel), {"field", "ref_id", "ref_collection", "ref_name"})

    def test_sensitive_fields_never_reach_embeddings(self):
        self.ingest()
        self.assertEqual(self.db["rag_embeddings"].count_documents({"text": {"$regex": "notarealhash|Password"}}), 0)

    def test_second_run_embeds_nothing_then_only_changes_and_deletions(self):
        first = self.ingest()
        total = sum(s["embedded"] for s in first["collections"].values())
        self.assertGreater(total, 10)
        second = self.ingest()
        self.assertEqual(sum(s["embedded"] for s in second["collections"].values()), 0)

        self.db["projects"].update_one({"project": "ShopEase"}, {"$set": {"status": "completed"}})
        self.db["tasks"].delete_one({"title": "Fix login bug"})
        third = self.ingest()
        self.assertEqual(third["collections"]["projects"]["embedded"], 1)
        self.assertEqual(third["collections"]["tasks"]["removed"], 1)
        self.assertEqual(self.db["rag_embeddings"].count_documents({"text": {"$regex": "Fix login bug"}}), 0)

    def test_renaming_a_referenced_document_refreshes_the_documents_that_point_to_it(self):
        self.ingest()
        self.db["workspaces"].update_one({"name": "Acme Corp"}, {"$set": {"name": "Acme Global"}})
        summary = self.ingest()
        self.assertGreaterEqual(summary["collections"]["memberships"]["embedded"], 1)
        self.assertEqual(self.db["rag_embeddings"].count_documents({"text": {"$regex": "Acme Global"}}) > 0, True)

    def test_full_mode_reembeds_everything(self):
        self.ingest()
        again = self.ingest(full=True)
        self.assertEqual(sum(s["unchanged"] for s in again["collections"].values()), 0)

    def test_unknown_collection_is_a_clean_error(self):
        from rag.services.exceptions import IngestionError
        with self.assertRaises(IngestionError):
            self.ingest(collections=["nope"])

    def test_embeddings_collection_is_never_ingested_itself(self):
        self.ingest()
        self.assertNotIn("rag_embeddings", ingestion_service.mongo_service.list_collections())


class SearchTests(MongoCase):
    def setUp(self):
        super().setUp()
        self.ingest()

    def test_vector_search_finds_the_right_record(self):
        hits = retrieval.search("Health4Travel completed", top_k=3, hybrid=False)
        self.assertTrue(any(h.collection == "projects" and h.display_name == "Health4Travel" for h in hits))
        self.assertTrue(all(h.vector_score is not None for h in hits))

    def test_collection_filter(self):
        hits = retrieval.search("Acme Corp", top_k=5, collections=["workspaces"], hybrid=False)
        self.assertTrue(hits and all(h.collection == "workspaces" for h in hits))

    def test_keyword_failure_falls_back_to_vector_search(self):
        hits = retrieval.search("Health4Travel completed", top_k=3, hybrid=True)  # mongomock has no $text
        self.assertTrue(hits)

    def test_rrf_fusion_rewards_agreement(self):
        fused = rrf_fuse([("a", .9), ("b", .8)], [("b", 5.0), ("c", 4.0)], k=3, keyword_weight=1.0, rrf_k=60)
        self.assertEqual(fused[0][0], "b")
        self.assertEqual({i for i, _ in fused}, {"a", "b", "c"})

    def test_multi_hop_follows_relationships_forward_and_reverse(self):
        hits = [h for h in retrieval.search("Rohan Mehta backend developer", top_k=8, hybrid=False)
                if h.collection == "users"][:1]
        self.assertEqual(hits[0].display_name, "Rohan Mehta")
        related = retrieval.expand(hits, hops=1, max_related=10)
        self.assertTrue(any(r.collection == "tasks" and r.via["direction"] == "reverse" for r in related))

        task = [h for h in retrieval.search("Fix login bug", top_k=8, hybrid=False) if h.collection == "tasks"][0]
        forward = {(r.collection, r.display_name) for r in retrieval.expand([task], hops=1, max_related=10)
                   if r.via["direction"] == "forward"}
        self.assertIn(("projects", "ShopEase"), forward)
        self.assertIn(("users", "Rohan Mehta"), forward)

    def test_two_hops_reach_the_workspace_through_the_project(self):
        task = [h for h in retrieval.search("Fix login bug", top_k=8, hybrid=False) if h.collection == "tasks"][0]
        related = retrieval.expand([task], hops=2, max_related=10)
        self.assertIn("workspaces", {r.collection for r in related})


class AskTests(MongoCase):
    def setUp(self):
        super().setUp()
        self.ingest()

    def test_ask_sends_context_with_related_records_and_returns_sources(self):
        with patch.object(llm_service, "generate", return_value="ShopEase is in progress.") as gen:
            result = rag_service.ask("What is the status of ShopEase?", hops=1)
        self.assertEqual(result["answer"], "ShopEase is in progress.")
        context = gen.call_args[0][1]
        self.assertIn("ShopEase", context)
        self.assertIn("[1]", context)
        self.assertTrue(any(s["collection"] == "projects" for s in result["sources"]))
        self.assertIn("search", result["timings_ms"])

    def test_not_found_answer_has_no_sources(self):
        with patch.object(llm_service, "generate", return_value=llm_service.NOT_FOUND_ANSWER):
            self.assertEqual(rag_service.ask("What is the CEO's phone number?")["sources"], [])

    def test_answers_are_cached_until_the_index_changes(self):
        with patch.object(llm_service, "generate", return_value="ok") as gen:
            rag_service.ask("Which projects are completed?")
            second = rag_service.ask("which   projects are COMPLETED?")
            self.assertTrue(second["cached"])
            self.assertEqual(gen.call_count, 1)
            self.db["projects"].update_one({"project": "ShopEase"}, {"$set": {"status": "paused"}})
            self.ingest()
            rag_service.ask("Which projects are completed?")
            self.assertEqual(gen.call_count, 2)

    @override_settings(MAX_CONTEXT_CHARS=300)
    def test_context_respects_the_character_budget(self):
        with patch.object(llm_service, "generate", return_value="ok") as gen:
            rag_service.ask("Tell me about tasks and projects", top_k=10)
        self.assertLess(len(gen.call_args[0][1]), 900)

    def test_validation(self):
        for bad in ("  ", "x" * 501, 5):
            with self.assertRaises(InvalidRequestError):
                rag_service.validate_question(bad)

    def test_missing_index_is_a_clean_error(self):
        self.db["rag_embeddings"].delete_many({})
        vector_index.reset_for_tests()
        import shutil
        from django.conf import settings
        shutil.rmtree(settings.VECTOR_STORE_DIR, ignore_errors=True)
        with self.assertRaises(VectorStoreError):
            rag_service.ask("anything at all")

    @override_settings(LLM_API_KEY="")
    def test_missing_llm_key_is_a_clean_error(self):
        from rag.services.exceptions import ConfigError
        llm_service._client = None
        with self.assertRaises(ConfigError):
            rag_service.ask("Which projects are completed?")


class ApiTests(MongoCase):
    def setUp(self):
        super().setUp()
        self.client = Client()

    def post(self, url, payload, **extra):
        return self.client.post(url, json.dumps(payload), content_type="application/json", **extra)

    def test_index_page(self):
        self.assertContains(self.client.get("/"), "Ask your database")

    def test_ask_endpoint_end_to_end(self):
        self.ingest()
        with patch.object(llm_service, "generate", return_value="Health4Travel is completed."):
            response = self.post("/api/ask/", {"question": "Which projects have been finished?"})
        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["answer"], "Health4Travel is completed.")
        self.assertTrue(data["sources"])

    def test_search_endpoint(self):
        self.ingest()
        data = self.post("/api/search/", {"query": "Acme Corp workspace", "top_k": 3, "hybrid": False,
                                          "expand": True}).json()
        self.assertLessEqual(len(data["results"]), 3)
        self.assertIn("related", data)
        self.assertIn("relationships", data["results"][0])

    def test_bad_input_returns_json_errors(self):
        self.assertEqual(self.client.post("/api/ask/", "nope", content_type="application/json").status_code, 400)
        self.assertEqual(self.post("/api/ask/", {"question": "hello", "top_k": 999}).status_code, 400)
        self.assertEqual(self.post("/api/search/", {"query": "x", "collections": "users"}).status_code, 400)
        self.assertEqual(self.client.get("/api/ask/").status_code, 405)

    def test_ask_before_ingest_explains_what_to_do(self):
        response = self.post("/api/ask/", {"question": "Which projects are completed?"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("ingest", response.json()["error"])

    @override_settings(INGEST_TOKEN="s3cret")
    def test_ingest_requires_the_token(self):
        self.assertEqual(self.post("/api/ingest/", {}).status_code, 401)
        self.assertEqual(self.client.get("/api/ingest/", HTTP_AUTHORIZATION="Bearer wrong").status_code, 401)
        with patch("rag.views.ingestion_service.start_job", return_value={"id": "x", "state": "running"}):
            response = self.post("/api/ingest/", {"full": True}, HTTP_AUTHORIZATION="Bearer s3cret")
        self.assertEqual(response.status_code, 202)

    @override_settings(INGEST_TOKEN="", DEBUG=False)
    def test_ingest_is_disabled_without_a_token_in_production(self):
        self.assertEqual(self.post("/api/ingest/", {}).status_code, 401)

    def test_health(self):
        self.ingest()
        data = self.client.get("/api/health/").json()
        self.assertEqual(data["status"], "ok")
        self.assertGreater(data["embeddings"], 0)
        self.assertTrue(data["index"]["ready"])
