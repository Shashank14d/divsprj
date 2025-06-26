# 🎯 Smart AI-Based Diet and Workout Planner

A comprehensive full-stack AI-powered fitness application that generates personalized diet and workout plans based on user inputs, BMI analysis, and AI-powered body type classification.

## 🌟 Features

### 🤖 AI-Powered Analysis
- **CNN Body Type Detection**: Uses Convolutional Neural Networks to classify body types (Ectomorph, Mesomorph, Endomorph)
- **BMI Calculation**: Automatic BMI calculation and categorization
- **Personalized Recommendations**: AI-driven diet and workout suggestions

### 📊 Data Processing
- **Excel Integration**: Handles real-time dataset from Google Forms
- **Data Validation**: Comprehensive data validation and cleaning
- **Statistical Analysis**: Detailed analytics and insights

### 🎨 Modern UI/UX
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Interactive Forms**: Real-time validation and feedback
- **Beautiful Animations**: Smooth transitions and modern aesthetics
- **Image Upload**: Drag-and-drop image functionality

### 🍽️ Personalized Diet Plans
- **Food Preference Support**: Vegetarian, Non-Vegetarian, and Vegan options
- **Goal-Based Planning**: Weight Loss, Muscle Gain, and Stay Fit plans
- **BMI-Conscious**: Tailored recommendations based on BMI category

### 💪 Custom Workout Routines
- **7-Day Plans**: Complete weekly workout schedules
- **Fitness Level Adaptation**: Adjusts intensity based on BMI and goals
- **Exercise Variety**: Mix of strength training, cardio, and flexibility

## 🛠️ Technology Stack

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Modern styling with animations
- **Bootstrap 5**: Responsive framework
- **JavaScript**: Interactive functionality
- **Font Awesome**: Beautiful icons

### Backend
- **Flask**: Python web framework
- **TensorFlow/Keras**: Deep learning for body type classification
- **OpenCV**: Image processing and analysis
- **Pandas**: Data manipulation and Excel handling

### Data & ML
- **scikit-learn**: Machine learning utilities
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization

## 📁 Project Structure

```
divsprj/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── data_processor.py              # Data processing utilities
├── models/
│   ├── train_model.py             # CNN model training script
│   └── body_type_model.h5         # Trained model (generated)
├── templates/
│   ├── index.html                 # Main form page
│   └── results.html               # Results display page
├── static/
│   ├── css/
│   │   └── style.css              # Custom styles
│   ├── js/
│   │   └── script.js              # Interactive JavaScript
│   └── uploads/                   # User uploaded images
├── training_data/                 # Processed training data
└── README.md                      # This file
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd divsprj
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up Directories
```bash
# Create necessary directories
mkdir -p static/uploads
mkdir -p models
mkdir -p training_data
```

### Step 5: Train the Model (Optional)
```bash
python models/train_model.py
```

### Step 6: Run the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## 📋 Usage Guide

### 1. User Input Collection
- **Personal Information**: Name, age, gender
- **Physical Measurements**: Height (feet/inches), weight (kg)
- **Preferences**: Food preference (Veg/Non-Veg/Vegan)
- **Goals**: Fitness objective (Weight Loss/Muscle Gain/Stay Fit)
- **Health Information**: Optional health issues
- **Body Image**: Full-body photo upload

### 2. AI Analysis Process
1. **BMI Calculation**: Automatic computation from height and weight
2. **BMI Categorization**: Underweight, Normal, Overweight, Obese
3. **Body Type Detection**: CNN analysis of uploaded image
4. **Plan Generation**: Personalized diet and workout recommendations

### 3. Results Display
- **Personal Profile**: User information and analysis results
- **Diet Plan**: 5-meal daily plan with specific food recommendations
- **Workout Plan**: 7-day exercise routine with detailed instructions
- **AI Recommendations**: Health tips and progress tracking advice

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:
```env
FLASK_SECRET_KEY=your-secret-key-here
UPLOAD_FOLDER=static/uploads
MAX_CONTENT_LENGTH=10485760
```

### Model Configuration
Edit `models/train_model.py` to customize:
- Model architecture
- Training parameters
- Data preprocessing
- Model evaluation metrics

## 📊 Data Processing

### Excel Data Format
The application expects Excel files with the following columns:
- Name, Age, Gender
- Height_Feet, Height_Inches, Weight_KG
- Food_Preference, Fitness_Goal
- Health_Issues (optional)
- Image_Path (optional)

### Data Processing Commands
```bash
# Process Excel data
python data_processor.py

# Export for training
python -c "from data_processor import DataProcessor; DataProcessor().export_for_training()"
```

## 🤖 Machine Learning Model

### CNN Architecture
- **Input**: 224x224x3 RGB images
- **Convolutional Layers**: 4 layers with increasing filters (32, 64, 128, 256)
- **Pooling**: MaxPooling2D after each conv layer
- **Regularization**: BatchNormalization and Dropout
- **Output**: 3 classes (Ectomorph, Mesomorph, Endomorph)

### Training Process
1. **Data Preparation**: Image preprocessing and augmentation
2. **Model Training**: 50 epochs with early stopping
3. **Validation**: 20% validation split
4. **Model Saving**: Best model saved as HDF5 file

## 🎨 Customization

### Styling
Edit `static/css/style.css` to customize:
- Color scheme
- Typography
- Animations
- Responsive breakpoints

### Templates
Modify `templates/` files to change:
- Form layout
- Results display
- Navigation structure

### JavaScript
Update `static/js/script.js` for:
- Form validation
- Image upload behavior
- Interactive features

## 📱 Responsive Design

The application is fully responsive and optimized for:
- **Desktop**: Full-featured experience
- **Tablet**: Touch-friendly interface
- **Mobile**: Streamlined mobile layout

## 🔒 Security Features

- **File Upload Validation**: Type and size restrictions
- **Input Sanitization**: XSS protection
- **Secure File Handling**: Safe file storage
- **Form Validation**: Client and server-side validation

## 📈 Performance Optimization

- **Image Compression**: Automatic resizing and optimization
- **Lazy Loading**: Efficient resource loading
- **Caching**: Static asset caching
- **Database Optimization**: Efficient data queries

## 🧪 Testing

### Manual Testing
1. Test form submission with various inputs
2. Verify image upload functionality
3. Check responsive design on different devices
4. Validate BMI calculations
5. Test plan generation accuracy

### Automated Testing (Future Enhancement)
```bash
# Run tests (when implemented)
python -m pytest tests/
```

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker (Dockerfile to be created)
docker build -t fitness-planner .
docker run -p 5000:5000 fitness-planner
```

## 📝 API Endpoints

### Main Routes
- `GET /`: Main form page
- `POST /submit`: Form submission and plan generation
- `POST /api/calculate_bmi`: BMI calculation API

### Response Format
```json
{
  "bmi": 24.5,
  "bmi_category": "Normal",
  "body_type": "Mesomorph",
  "diet_plan": {...},
  "workout_plan": {...}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Bootstrap**: For the responsive framework
- **Font Awesome**: For beautiful icons
- **TensorFlow**: For machine learning capabilities
- **OpenCV**: For image processing
- **Flask**: For the web framework

## 📞 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation

## 🔮 Future Enhancements

- **User Authentication**: Login and profile management
- **Progress Tracking**: Weight and measurement logging
- **Social Features**: Community and sharing
- **Mobile App**: Native iOS/Android applications
- **Advanced Analytics**: Detailed progress reports
- **Integration**: Fitness tracker and nutrition app APIs

---

**Made with ❤️ for better health and fitness** 