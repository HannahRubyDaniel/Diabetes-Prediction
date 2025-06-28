from django.db import models
from django.db.models import FloatField
from django.utils import timezone
from django.contrib.auth.models import User  # Import Django's User model

class PredictionHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="predictions")
    pregnancies = models.IntegerField()
    glucose = models.FloatField()
    blood_pressure = models.FloatField()
    skin_thickness = models.FloatField()
    insulin = models.FloatField()
    bmi = models.FloatField()
    diabetes_pedigree_function = models.FloatField()
    age = models.IntegerField()
    prediction = models.CharField(max_length=20, choices=[("Positive", "Positive"), ("Negative", "Negative")], default="Not Predicted")
    created_at = models.DateTimeField(auto_now_add=True)


    def __str__(self):
        return f"{self.user.username} - {self.prediction} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"
