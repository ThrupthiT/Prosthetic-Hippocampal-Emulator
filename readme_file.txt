steps
1. Create virtual environment
cd C:\projects\prosthetic_hippocampal_using_cnn
python -m venv cnn_env
cnn_env\Scripts\activate

2. Libraries
pip install numpy pandas matplotlib mne pillow scikit-learn joblib
pip install torch torchvision
pip install sentence-transformers

3. Load and process EEG
python src/load_and_preprocess_eeg.py
Output:
	outputs/
		Subject00_1.png
		Subject00_2.png
		Subject01_1.png
		....
4. Feature Extraction
python src/feature_extraction.py
Output: 
	features/eeg_band_features.csv

5. prepare CNN Dataset
python src/prepare_cnn_dataset.py
Output:
	cnn_dataset/
		encode/
		recall/

6.Train CNN model
python src/train_pytorch_cnn.py
Output:
	outputs/memory_cnn.pth

7.Run memory assistant
python app_logic/memory_app.py

* test_samples/test1.png and test_samples/test2.png are encode signals
* remaining are decode signals

***EXPLAINATION
When the EEG image is classified as an encoding state, the system assumes the brain is actively trying to store something important, so it asks the user what they want to remember and at what time. 
This information is stored along with a semantic representation of the text, similar to how the hippocampus binds meaning and context together.
Later, when a new EEG image is classified as a recall state, the system does not blindly show all reminders. 
Instead, it looks at all stored memories and compares them based on how close they are in meaning to the current moment and recall intent using the embeddings. 
The memory that best matches this internal “competition” is retrieved and shown to the user. 
In this way, the system mimics hippocampal behavior by triggering memory storage and retrieval based on brain-state signals rather than fixed alarms, making it fundamentally different from a normal reminder application.