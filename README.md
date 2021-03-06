# torso-detection

Skin color classification model trained on FairFace dataset.
Download [the dataset](https://drive.google.com/file/d/1YtY4Zx5Xx0kpAeyKEEHoAlTpooFa04Xf/view?usp=sharing) and place it in the folder data/fairface

To train the network, run:

    python3 train.py --config config/base.yaml
    
To inference a single image, run:

    python3 inference.py --config config/base.yaml --model-path /path/to/model.pt --image-path samples/test1.jpg
    
Inference results will be saved to inference/

Pretrained model: [model.pt](https://drive.google.com/file/d/1eJbJg2uya6aq1whU7VZyYSRRQAj-w015/view?usp=sharing)

Examples:
[test1.jpg](inference/inference_test1.jpg)
[test2.jpg](inference/inference_test2.jpg)
