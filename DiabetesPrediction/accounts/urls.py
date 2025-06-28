from django.urls import path
from . import views
from DiabetesPrediction import views as main_views

urlpatterns = [
    path("", main_views.home, name="home"),
    path("learn_about_diabetes/", main_views.learn_about_diabetes, name="learn_about_diabetes"),
    path('guidelines/', main_views.guidelines, name='guidelines'),
    path("login/", main_views.user_login, name="login"),
    path("logout/", main_views.user_logout, name="logout"),
    path("predict/", main_views.predict, name="predict"),
    path("history/", main_views.history, name="history"),
    path("predict/result/", main_views.result, name="result"),
    path("register/", main_views.user_register, name="register")
]
