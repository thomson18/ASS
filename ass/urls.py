from django.conf.urls import url
from django.urls import path
from . import views

urlpatterns = [
    path('', views.hp, name='hp'),
    path('about/', views.about, name='about'),
    path('features/', views.features, name='features'),
    path('contact/', views.contact, name='contact'),
    path('services/', views.services, name='services'),
    url(r'^/(?P<stream_path>(.*?))/$',views.dynamic_stream,name="videostream"),
    url(r'^stream/$',views.indexscreen),
    path('login/', views.login, name='login'),
    path('logout/', views.logout, name='logout'),
    path('signup/', views.signup, name='signup'),
    path('fetcher/',views.datafetcher, name='fetcher'),
    path('trainer/', views.trainer, name='trainer'),
    path('rekognizer/', views.rekognizer, name='rekognizer'),
    path('security/', views.security, name='security'),
    path('attendance/', views.attendance, name='attendance'),
    path('det/', views.det, name='det'),
]