#!/usr/bin/env python3
import zipfile

with zipfile.ZipFile("task_ABCD_D_filtered.zip", 'r') as zip_ref:
    zip_ref.extractall("task_ABCD_D_filtered")