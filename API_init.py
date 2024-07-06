#init function takes config file as input, returns the required model
import yaml
import os
def init_model(config):
#read yaml file
    with open(config,'r') as file:
        config_file = yaml.safe_load(file)
#print successful message and the config file
    print("*"*40)
    print("Successfully loaded config file!")
    #post-process config file
    config_file['camera_names'] = config_file['camera_names'].split(' ')
    config_file['dataset_dir'] = os.path.expanduser(config_file['data_dir'])+ config_file['dataset_dir']
    config_file['ckpt_dir'] = os.path.expanduser(config_file['data_dir']) + config_file['ckpt_dir'] 
    for item in config_file:
        print(item,":",config_file[item])
    print("*"*40)
    return config_file


def main():
    init_model('transfer_straw.yaml')

if __name__ == "__main__":
    main()