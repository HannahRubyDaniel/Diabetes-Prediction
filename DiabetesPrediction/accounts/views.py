from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages


def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists")
        else:
            user = User.objects.create_user(username=username, password=password)
            user.save()
            messages.success(request, "Account created. Please log in.")
            return redirect('login')
    return render(request, 'accounts/register.html')


def user_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('predict')  # Redirect to prediction
        else:
            messages.error(request, "Invalid credentials")
    return render(request, 'accounts/login.html')


def user_logout(request):
    logout(request)
    return redirect('login')
