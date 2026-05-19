#!/usr/bin/env python3
import zipfile

with zipfile.ZipFile("task_ABC_D.zip", 'r') as zip_ref:
    zip_ref.extractall("task_ABC_D")