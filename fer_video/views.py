from django.shortcuts import render, redirect
from django.conf import settings
from .forms import VideoForm
from .models import Video
from .utils import *
from .extract_frames import extract_frames
from .modules.ST_Former import GenerateModel 
from .modules.datasets import VideoDataset
import os
import torch
import json

model = GenerateModel().cuda()
model.eval()
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load(os.path.join(settings.STATICFILES_DIRS[0], 'checkpoints', '05-25-18_18-model_best.pth'))
model.load_state_dict(checkpoint['state_dict'])


def home(request):
    # Define the path to the video directory
    video_dir = os.path.join('static', 'videos')
    
    # List all video files in the directory
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    # Get the latest uploaded video if available
    latest_video = Video.objects.order_by('-id').first()
    
    form = VideoForm()
    
    return render(request, 'home.html', {'video_files': video_files, 'latest_video': latest_video, 'form': form})



def upload_video(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_video = form.save()
            print(f"Video saved: {uploaded_video}")  # Debug print
            return redirect('home')
        else:
            print(f"Form errors: {form.errors}")  # Debug print
    else:
        form = VideoForm()
    
    latest_video = Video.objects.order_by('-id').first()
    print(f"Latest video: {latest_video}")  # Debug print
    return render(request, 'upload_video.html', {'form': form, 'latest_video': latest_video})


def analyze_video(request):
    dict_map_label = {0:"Happy", 1:"Sad", 2:"Neutral", 3:"Angry", 4:"Surprise", 5:"Disgust", 6:"Fear"}
    if request.method == 'POST':
        video_url = request.POST.get('video_url')
        print("analyze_video:", video_url)
        if '/static/videos/' in video_url:
            frames_url = '/static/frames/' + video_url.split('/')[-1].split('.')[0].split('_')[-1]
        else:
            frames_url = video_url
            print("analyze_video:", extract_frames(frames_url))
            
        pred_1, pred_2, ip1, ip2, attn, attn2, inp_idx, x_sample_idx = run_prediction(frames_url)
        vectors = [ip1, ip2]
        images = tensors_to_base64_images(vectors, [inp_idx, x_sample_idx])
        pred_2 = pred_2.softmax(-1)
        pred_1 = pred_1.softmax(-1)
        pred_lb_1 = pred_1.argmax(-1).item()
        pred_lb_2 = pred_2.argmax(-1).item()
        return render(request, 'analyze_video.html', {'video_url': video_url, 
                                                      'prediction_label_1': dict_map_label[pred_lb_1], 
                                                      'prediction_label_2': dict_map_label[pred_lb_2], 
                                                      'stage1_dt': json.dumps(pred_1[0].tolist()), 
                                                      'stage2_dt': json.dumps(pred_2[0].tolist()), 
                                                      'attn' : json.dumps(attn[0][0].tolist()),
                                                      'attn2' : json.dumps(attn2[0][0].tolist()),
                                                      'images':images})

def run_prediction(video_url):
    # This function should contain the logic to run the prediction model on the uploaded video
    # For simplicity, we'll return a dummy result here
    if '/static/frames/' in video_url:
        print(settings.STATICFILES_DIRS[0], type(settings.STATICFILES_DIRS[0]))
        print(video_url.split('/')[-1], type(video_url.split('/')[-1]))
        frames_path = os.path.join(settings.STATICFILES_DIRS[0], 'frames', video_url.split('/')[-1])
    else:
        frames_path = os.path.join(settings.MEDIA_ROOT, 'frames')
    print("DEBUG run_prediction:", frames_path)
    print("DEBUG run_prediction:", os.path.dirname(os.path.dirname(__file__)))
    print("DEBUG run_prediction:", os.path.exists(frames_path))
    print("DEBUG run_prediction:", os.path.exists(os.path.join(os.path.dirname(__file__), frames_path)))
    
    data = VideoDataset(frames_path)
    inp = data.__getitem__(0)
    print("DEBUG run_prediction:", inp.shape)
    out_1, out_2, input, x_sample, attn, attn2 = model(inp)
    inp_idx, x_sample_idx = get_index(inp, input[0].permute(1,0,2,3), x_sample[0].permute(1,0,2,3))
    return out_1, out_2, input, x_sample, attn, attn2, inp_idx, x_sample_idx


