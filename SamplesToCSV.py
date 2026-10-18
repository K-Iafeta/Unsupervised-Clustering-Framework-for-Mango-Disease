#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import hashlib
import pandas as pd

root_dir = './Images'  # Change to your images root
manifest = []
duplicates = []

for subdir, _, files in os.walk(root_dir):
    print(subdir)
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(subdir, file)
            manifest.append({'path': path})

df = pd.DataFrame(manifest)


# In[10]:


category_names = []
class_names = []
plant_part_names = []
for idx in df.index:
    category_name = df.loc[idx, 'path'].split('/')[2]
    category_names.append('_'.join(category_name.split('_')[1:]))

    class_name = df.loc[idx, 'path'].split('/')[3]
    class_names.append('_'.join(class_name.split('_')[1:]))

    plant_part_name = df.loc[idx, 'path'].split('/')[4]
    plant_part_names.append('_'.join(plant_part_name.split('_')[1:]))


# In[11]:


print(len(list(set(class_names))))
list(set(class_names))


# In[12]:


#
print(len(list(set(category_names))))
list(set(category_names))


# In[13]:


print(len(list(set(plant_part_names))))
list(set(plant_part_names))


# In[14]:


df['Category'] = category_names


# In[15]:


df['Class'] = class_names


# In[16]:


df['Plant_Part'] = plant_part_names


# In[17]:


df


# In[11]:


df.to_csv('MangoImageSample.csv')


# In[ ]:




