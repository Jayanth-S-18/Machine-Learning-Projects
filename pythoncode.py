import os
import numpy as np
import joblib
import gc
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

!unzip -q dataset.zip

BASE_PATH = '/content/' 
STAGES = ['cvmi 1', 'cvmi 2', 'cvmi 3', 'cvmi stage 4', 'cvmi stage 5', 'cvmi 6']
IMG_SIZE = (224, 224)

def get_optimized_features():
    extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    X_list = []
    y_list = []
    
    for i, stage in enumerate(STAGES):
        folder = os.path.join(BASE_PATH, stage)
        if not os.path.exists(folder): continue
        
        for img_name in os.listdir(folder):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = load_img(os.path.join(folder, img_name), target_size=IMG_SIZE)
                img_arr = img_to_array(img)
                img_expanded = np.expand_dims(img_arr, 0)
                
                count = 0
                for batch in datagen.flow(img_expanded, batch_size=1):
                    processed = preprocess_input(batch[0])
                    feature = extractor.predict(np.expand_dims(processed, 0), verbose=0)
                    X_list.append(feature.flatten())
                    y_list.append(i)
                    count += 1
                    if count >= 30: break
                del img, img_arr, img_expanded
        gc.collect()
        
    return np.array(X_list), np.array(y_list)

X_features, y_labels = get_optimized_features()

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_labels, test_size=0.2, stratify=y_labels, random_state=42
)

param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [10, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1), 
    param_grid, cv=3, return_train_score=True
)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
train_acc = grid_search.cv_results_['mean_train_score'][grid_search.best_index_]
test_acc = best_rf.score(X_test, y_test)

print(f"Train Score: {train_acc:.4f}")
print(f"Test Score: {test_acc:.4f}")

joblib.dump(best_rf, 'cvm_optimized_rf.pkl')
