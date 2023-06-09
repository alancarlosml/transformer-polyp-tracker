# Generated by Django 4.1.7 on 2023-03-29 21:14

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='VideoProcessing',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file_type', models.IntegerField()),
                ('filename', models.CharField(max_length=100)),
                ('start_time', models.DateTimeField(auto_now_add=True)),
                ('end_time', models.DateTimeField(null=True)),
                ('status', models.CharField(max_length=100)),
            ],
        ),
    ]
