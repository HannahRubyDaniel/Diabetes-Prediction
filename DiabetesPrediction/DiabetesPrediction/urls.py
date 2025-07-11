"""
URL configuration for DiabetesPrediction project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from DiabetesPrediction import views  # Import views correctly

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.home, name="home"),  # Ensure home is correctly linked
    path("Learn_about_diabetes/", views.learn_about_diabetes, name="learn_about_diabetes"),
    path('guidelines/', views.guidelines, name='guidelines'),
    path("predict/", views.predict, name="predict"),
    path("predict/result/", views.result, name="result"),
    path("history/", views.history, name="history"),
    path("login/", views.user_login, name="login"),
    path("logout/", views.user_logout, name="logout"),
    path("register/", views.user_register, name="register"),
    path("accounts/", include("accounts.urls")),  # Ensure accounts app is included
]

