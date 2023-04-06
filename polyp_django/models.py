from django.db import models
import uuid

class VideoProcessing(models.Model):
    id = models.AutoField(primary_key=True, editable=False)
    file_type = models.IntegerField(null=False)
    filename = models.CharField(max_length=100)
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True)
    status = models.CharField(max_length=100)

    class Meta:
        app_label = 'polyp_django'
