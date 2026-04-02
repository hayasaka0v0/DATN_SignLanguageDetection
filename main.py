# import os # Thư viện tương tác với hệ điều hành



# list_video = os.listdir("Projects/DATN/DATN_SignLanguageDetection/data/raw_videos/")
# for folder in list_video:
#     print("Title: " + folder)
#     print(os.listdir("Projects/DATN/DATN_SignLanguageDetection/data/raw_videos/" + folder))
#     print("\n")


import os # Thư viện tương tác với hệ điều hành

base_path = "Projects/DATN/DATN_SignLanguageDetection/data/raw_videos/"

list_video = os.listdir(base_path)
for folder in list_video:
    print("Title: " + folder)
    full_path = os.path.join(base_path, folder)
    for video in os.listdir(full_path):
        if video.endswith(".mp4") or video.endswith(".webm"):
            print(video)
    print("\n")
