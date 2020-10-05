import os
import socket

res_dict = {
            'qVGA' : '320x240',
            'VGA' : '640x480',
            'nHD' : '640x360',
            'qHD' : '960x540',
            'HD' : '1280x720',
            'FHD' : '1920x1080',
            'Others' : '320x240'
        }

class get_args_and_set_stream:
    def __init__(self, args_dict):
        self.argdict = args_dict
    
    def convert_res(self, res):
        try:
            return res_dict[res]
        except:
            return res_dict['qVGA']

    def write_setting_to_file(self):
        if not self.argdict["multi"]:
            command = "mjpg_streamer -i \"input_uvc.so -f " + self.argdict["fps"] + " -r " + self.convert_res(self.argdict["res"]) + "\" "
            command += "-o \"output_http.so -p " + self.argdict["port"] + " -w /usr/local/share/mjpg-streamer/www/\""

            with open("./cam/cam_streaming.sh", 'w') as f:
                f.write(command)

        else:
            print("멀티캠은 미구현 ㅎㅎ")


class main_discription:
    def __init__(self, args_dict):
        self.argdict = args_dict

    def print_info(self):
        for item in self.argdict.items():
            print(item[0], " : ", item[1])
    
    def print_res(self):
        for item in res_dict.items():
            print(item[0], " : ", item[1])
    
    def print_main(self):
        print("\n\n#########################################")
        print("#### CAMERA STREAMING SERVER RUNNING ####")
        print("## res  :  for see detail resolutions  ##")
        print("## info :  for see streaming infos     ##")
        print("## open :  for open streaming webpages ##")
        print("## quit :  for shutdown stream server  ##")
        print("## ping :  for check connect streaming ##")
        print("#########################################")


