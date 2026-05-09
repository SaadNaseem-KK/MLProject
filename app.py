from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import io
import base64
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Model Architecture (from your code)
class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, input):
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))
        
        init = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1)
        if xb.is_cuda:
            init = init.cuda()
        matrix = self.fc3(xb).view(-1, self.k, self.k) + init
        return matrix


class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, input):
        matrix3x3 = self.input_transform(input)
        xb = torch.bmm(torch.transpose(input, 1, 2), matrix3x3).transpose(1, 2)
        
        xb = F.relu(self.bn1(self.conv1(xb)))
        
        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb, 1, 2), matrix64x64).transpose(1, 2)
        
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64


class PointNet(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        xb, matrix3x3, matrix64x64 = self.transform(input)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return self.logsoftmax(output), matrix3x3, matrix64x64


# Class names from ModelNet10
CLASS_NAMES = {
    0: 'bathtub',
    1: 'bed',
    2: 'chair',
    3: 'desk',
    4: 'dresser',
    5: 'monitor',
    6: 'night_stand',
    7: 'sofa',
    8: 'table',
    9: 'toilet'
}

# Initialize model
device = torch.device("cpu")  # Use CPU for deployment
model = PointNet(classes=10)
model.eval()

# Try to load pre-trained weights if available
try:
    model.load_state_dict(torch.load('save.pth', map_location=device))
    print("Model loaded successfully!")
except:
    print("No pre-trained model found. Using random weights for demo.")


def read_off(file_content):
    """Read OFF file format"""
    lines = file_content.decode('utf-8').strip().split('\n')
    if 'OFF' not in lines[0]:
        raise ValueError('Not a valid OFF header')
    
    # Parse header
    header = lines[1].strip().split()
    n_verts = int(header[0])
    n_faces = int(header[1])
    
    # Parse vertices
    verts = []
    for i in range(2, 2 + n_verts):
        vert = [float(x) for x in lines[i].strip().split()]
        verts.append(vert)
    
    # Parse faces
    faces = []
    for i in range(2 + n_verts, 2 + n_verts + n_faces):
        face = [int(x) for x in lines[i].strip().split()][1:]
        faces.append(face)
    
    return np.array(verts), faces


def sample_points(verts, faces, n_points=1024):
    """Sample points uniformly from mesh surface"""
    import random
    
    def triangle_area(pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5
    
    def sample_point(pt1, pt2, pt3):
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return np.array([f(0), f(1), f(2)])
    
    # Calculate areas
    areas = []
    for face in faces:
        area = triangle_area(verts[face[0]], verts[face[1]], verts[face[2]])
        areas.append(area)
    
    # Sample faces based on area
    sampled_faces = random.choices(faces, weights=areas, k=n_points)
    
    # Sample points
    sampled_points = np.zeros((n_points, 3))
    for i, face in enumerate(sampled_faces):
        sampled_points[i] = sample_point(verts[face[0]], verts[face[1]], verts[face[2]])
    
    return sampled_points


def normalize_pointcloud(pointcloud):
    """Normalize point cloud to unit sphere"""
    norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
    norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))
    return norm_pointcloud


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.off'):
            return jsonify({'error': 'Only .OFF files are supported'}), 400
        
        # Read file
        file_content = file.read()
        verts, faces = read_off(file_content)
        
        # Sample and normalize points
        pointcloud = sample_points(verts, faces, n_points=1024)
        pointcloud = normalize_pointcloud(pointcloud)
        
        # Prepare for model
        pointcloud_tensor = torch.from_numpy(pointcloud).float().unsqueeze(0)
        pointcloud_tensor = pointcloud_tensor.transpose(1, 2)
        
        # Predict
        with torch.no_grad():
            output, _, _ = model(pointcloud_tensor)
            probabilities = torch.exp(output)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get all class probabilities
        all_probs = {CLASS_NAMES[i]: float(probabilities[0][i]) for i in range(10)}
        
        # Prepare point cloud data for visualization
        pointcloud_data = {
            'x': pointcloud[:, 0].tolist(),
            'y': pointcloud[:, 1].tolist(),
            'z': pointcloud[:, 2].tolist()
        }
        
        return jsonify({
            'success': True,
            'predicted_class': CLASS_NAMES[predicted_class],
            'confidence': confidence,
            'all_probabilities': all_probs,
            'pointcloud': pointcloud_data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
