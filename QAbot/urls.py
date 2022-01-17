from django.urls import path
from . import views
urlpatterns=[
   path('',views.ready_view),
   path('reply/',views.reply),
]