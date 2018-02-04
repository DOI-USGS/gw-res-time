
# coding: utf-8

# In[ ]:

import os, sys
import flopy as fp


# In[ ]:

homes = ['../Models']
fig_dir = '../Figures'

mfpth = '../executables/MODFLOW-NWT_1.0.9/bin/MODFLOW-NWT_64.exe'
mp_exe_name = '../executables/modpath.6_0/bin/mp6.exe' 

dir_list = []
mod_list = []
i = 0

for home in homes:
    if os.path.exists(home):
        for dirpath, dirnames, filenames in os.walk(home):
            for f in filenames:
                if os.path.splitext(f)[-1] == '.nam':
                    mod = os.path.splitext(f)[0]
                    mod_list.append(mod)
                    dir_list.append(dirpath)
                    i += 1
print('    {} models read'.format(i))


# In[ ]:

for model_ws in dir_list:
    model = os.path.normpath(model_ws).split(os.sep)[2]
    nam_file = '{}.nam'.format(model)

    print("working model is {}".format(model_ws))

    fp.run_model(mfpth, nam_file, model_ws=model_ws, silent=False, pause=False, 
                 report=False, normal_msg='normal termination', async=False, cargs=None)


# In[ ]:



