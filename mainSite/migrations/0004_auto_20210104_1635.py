# Generated by Django 3.1.3 on 2021-01-04 13:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mainSite', '0003_auto_20201227_1230'),
    ]

    operations = [
        migrations.AlterField(
            model_name='person',
            name='history',
            field=models.CharField(blank=True, default=None, max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='person',
            name='id',
            field=models.AutoField(primary_key=True, serialize=False),
        ),
        migrations.AlterField(
            model_name='person',
            name='themes',
            field=models.CharField(blank=True, default=None, max_length=1000, null=True),
        ),
    ]
