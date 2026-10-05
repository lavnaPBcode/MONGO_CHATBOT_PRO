from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/ask/", views.ask, name="ask"),
    path("api/search/", views.search, name="search"),
    path("api/ingest/", views.ingest, name="ingest"),
    path("api/health/", views.health, name="health"),
]
