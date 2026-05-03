# PointNet-3D-Object-Classification-System
An end-to-end deep learning application that classifies 3D objects directly from point cloud data using the powerful PointNet architecture. This project showcases the intersection of computer vision, deep learning, and interactive web development.



![](point.gif)

🎯 Key Technical Achievements:

✅ Deep Neural Network Implementation

    • Custom PointNet architecture with T-Net spatial transformers
    • 3D feature extraction with 1024-dimensional global descriptors
    • Batch normalization and dropout for robust generalization
    • Multi-layer perceptron classifier for 10 object categories

✅ Advanced 3D Data Processing

    • OFF file format parsing and mesh processing
    • Uniform surface sampling (1024 points per model)
    • Point cloud normalization to unit sphere
    • Real-time geometric feature extraction

✅ Production-Ready Web Application

    • Flask backend with REST API endpoints
    • Interactive 3D visualization using Plotly
    • Real-time point cloud rendering with Three.js
    • Responsive UI with advanced animations and LiDAR scanning effects

✅ Performance Metrics

    • Fast inference: ~64ms per prediction
    • 10 object classes (furniture and household items)
    • Interactive 3D rotation and exploration
    • Real-time probability distribution visualization

💡 Why Point Clouds Matter:
Point clouds are fundamental in robotics, autonomous vehicles, AR/VR, and 3D scanning applications. Unlike traditional 2D images, point clouds preserve complete 3D spatial information, making them ideal for real-world perception tasks.
