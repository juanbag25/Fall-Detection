



def get_gyro_module(window):
    gyro_x = window['gyrox'].values
    gyro_y = window['gyroy'].values
    gyro_z = window['gyroz'].values
    gyro_module = (gyro_x**2 + gyro_y**2 + gyro_z**2) ** 0.5
    return gyro_module

# acceleration module and its stats

def get_acc_module(window):
    acc_x = window['accx'].values
    acc_y = window['accy'].values
    acc_z = window['accz'].values
    acc_module = (acc_x**2 + acc_y**2 + acc_z**2) ** 0.5
    return acc_module

def get_mean_acc_module(window):
    acc_module = get_acc_module(window)
    return acc_module.mean()

def get_std_acc_module(window):
    acc_module = get_acc_module(window)
    return acc_module.std()

def get_max_acc_module(window):
    acc_module = get_acc_module(window)
    return acc_module.max()

def get_min_acc_module(window):
    acc_module = get_acc_module(window)
    return acc_module.min()

##mean, std, max, min for accz, accy, accx

def get_mean_accz(window):
    values = window['accz'].values
    return values.mean()

def get_std_accz(window):
    values = window['accz'].values
    return values.std()

def get_max_accz(window):
    values = window['accz'].values
    return values.max()

def get_min_accz(window):   
    values = window['accz'].values
    return values.min()

def get_mean_accy(window):
    values = window['accy'].values
    return values.mean()

def get_std_accy(window):
    values = window['accy'].values
    return values.std()

def get_max_accy(window):
    values = window['accy'].values
    return values.max()

def get_min_accy(window):   
    values = window['accy'].values
    return values.min()

def get_mean_accx(window):  
    values = window['accx'].values
    return values.mean()

def get_std_accx(window):
    values = window['accx'].values
    return values.std()

def get_max_accx(window):
    values = window['accx'].values
    return values.max()

def get_min_accx(window):   
    values = window['accx'].values
    return values.min()

