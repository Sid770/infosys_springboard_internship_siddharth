{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from transformers import BertTokenizer, BertModel\n",
    "from gensim.models import Word2Vec\n",
    "import torch\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Loading the Prediction Data**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the dataset\n",
    "file_path = r\"B:\\OneDrive - Amity University\\Desktop\\Intern\\Infosys\\Prediction\\Copy of prediction_data (1).xlsx\"\n",
    "\n",
    "data = pd.read_excel(file_path)\n",
    "\n",
    "# Specify the columns for processing\n",
    "job_description_col = 'job_description'\n",
    "transcript_col = 'transcript'\n",
    "resume_col = 'resume'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Preproccessing  the data with bert and word2vec embedss**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\sidhe\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python311\\site-packages\\transformers\\tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "# Initialize BERT tokenizer and model\n",
    "bert_model_name = 'bert-base-uncased'\n",
    "tokenizer = BertTokenizer.from_pretrained(bert_model_name)\n",
    "bert_model = BertModel.from_pretrained(bert_model_name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# \n",
    "def get_bert_embeddings(text):\n",
    "    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)\n",
    "    with torch.no_grad():\n",
    "        outputs = bert_model(**inputs)\n",
    "    # Use the [CLS] token embedding as a summary of the text\n",
    "    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()\n",
    "    return cls_embedding\n",
    "\n",
    "# Process text columns with BERT embeddings\n",
    "def process_with_bert(column_name):\n",
    "    embeddings = []\n",
    "    for text in data[column_name].fillna(''):\n",
    "        embedding = get_bert_embeddings(text)\n",
    "        embeddings.append(embedding)\n",
    "    return embeddings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Generating BERT embeddings for job descriptions...\n",
      "Generating BERT embeddings for transcripts...\n",
      "Generating BERT embeddings for resumes...\n",
      "Preparing for Word2Vec embeddings...\n"
     ]
    }
   ],
   "source": [
    "print(\"Generating BERT embeddings for job descriptions...\")\n",
    "data[f'{job_description_col}_bert'] = process_with_bert(job_description_col)\n",
    "\n",
    "print(\"Generating BERT embeddings for transcripts...\")\n",
    "data[f'{transcript_col}_bert'] = process_with_bert(transcript_col)\n",
    "\n",
    "print(\"Generating BERT embeddings for resumes...\")\n",
    "data[f'{resume_col}_bert'] = process_with_bert(resume_col)\n",
    "\n",
    "# Prepare for Word2Vec embeddings\n",
    "print(\"Preparing for Word2Vec embeddings...\")\n",
    "text_data = data[[job_description_col, transcript_col, resume_col]].fillna('').values.flatten()\n",
    "tokenized_data = [text.split() for text in text_data]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "word2vec_model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)\n",
    "\n",
    "# Function to get Word2Vec embeddings\n",
    "def get_word2vec_embeddings(text):\n",
    "    words = text.split()\n",
    "    embeddings = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]\n",
    "    if embeddings:\n",
    "        return np.mean(embeddings, axis=0)\n",
    "    else:\n",
    "        return np.zeros(word2vec_model.vector_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "#\n",
    "def process_with_word2vec(column_name):\n",
    "    embeddings = []\n",
    "    for text in data[column_name].fillna(''):\n",
    "        embedding = get_word2vec_embeddings(text)\n",
    "        embeddings.append(embedding)\n",
    "    return embeddings\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Generating Word2Vec embeddings for job descriptions...\n",
      "Generating Word2Vec embeddings for transcripts...\n",
      "Generating Word2Vec embeddings for resumes...\n",
      "Processed data saved to C:\\Users\\sidhe\\Downloads\\processed_dataset_with_embeddings.xlsx\n"
     ]
    }
   ],
   "source": [
    "print(\"Generating Word2Vec embeddings for job descriptions...\")\n",
    "data[f'{job_description_col}_word2vec'] = process_with_word2vec(job_description_col)\n",
    "\n",
    "print(\"Generating Word2Vec embeddings for transcripts...\")\n",
    "data[f'{transcript_col}_word2vec'] = process_with_word2vec(transcript_col)\n",
    "\n",
    "print(\"Generating Word2Vec embeddings for resumes...\")\n",
    "data[f'{resume_col}_word2vec'] = process_with_word2vec(resume_col)\n",
    "\n",
    "# Save processed data\n",
    "output_file = r'C:\\Users\\sidhe\\Downloads\\processed_dataset_with_embeddings.xlsx'\n",
    "data.to_excel(output_file, index=False)\n",
    "print(f\"Processed data saved to {output_file}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Unnamed: 0.1  Unnamed: 0          ID           Name               Role  \\\n",
      "0           514         537  rivash0038    lahar singh  software engineer   \n",
      "1           214         225   benjry660  benjamin ryan      data engineer   \n",
      "2          1408        1467  rivash0968    amisha bedi     data scientist   \n",
      "3          1071        1122  rivash0623  kairav mishra    product manager   \n",
      "4           390         410   bradgr792  bradley gross    product manager   \n",
      "\n",
      "                                          transcript  \\\n",
      "0  **lahar singh: software engineer candidate int...   \n",
      "1  interview transcript: data engineer position\\n...   \n",
      "2  **interview transcript: amisha bedi, data scie...   \n",
      "3  **interview transcript: product manager positi...   \n",
      "4  product manager interview transcript\\n\\ninterv...   \n",
      "\n",
      "                                              resume  \\\n",
      "0  **lahar singh**\\n**software engineer candidate...   \n",
      "1  here's a sample resume for benjamin ryan apply...   \n",
      "2  **candidate profile: amisha bedi**\\n\\n**role:*...   \n",
      "3  **kairav mishra: product manager**\\n\\nas a sea...   \n",
      "4  here's a sample resume for bradley gross apply...   \n",
      "\n",
      "                                 Reason for decision  \\\n",
      "0  expected_experience : 9+ years, domains: e-com...   \n",
      "1                                       cultural fit   \n",
      "2  expected_experience : 6-8 years, domains: heal...   \n",
      "3  expected_experience : 6-8 years, domains: tech...   \n",
      "4                                       cultural fit   \n",
      "\n",
      "                                     job_description  num_words_in_transcript  \\\n",
      "0  communicated ideas clearly and effectively., h...                      956   \n",
      "1  we are looking for a skilled data engineer wit...                      551   \n",
      "2  lacked key technical skills for the role., nee...                      612   \n",
      "3  had impressive experience and qualifications.,...                      793   \n",
      "4  we are looking for a skilled product manager w...                      665   \n",
      "\n",
      "                                job_description_bert  \\\n",
      "0  [-0.299284279, 0.524214268, -0.3784495, -0.055...   \n",
      "1  [-0.0613610744, -0.328133166, -0.0494225807, -...   \n",
      "2  [-0.690594196, 0.0694749951, -0.567380726, -0....   \n",
      "3  [-0.276863039, 0.323161066, -0.300788939, -0.1...   \n",
      "4  [-0.222035304, -0.125937596, -0.136251494, -0....   \n",
      "\n",
      "                                     transcript_bert  \\\n",
      "0  [-0.104561403, -0.408662647, -0.37219432, -0.0...   \n",
      "1  [-0.538317442, -0.160756737, -0.559148788, 0.0...   \n",
      "2  [-0.267303348, -0.507924497, -0.403460264, 0.1...   \n",
      "3  [-0.717701018, -0.267151356, -0.680159688, -0....   \n",
      "4  [-0.349911988, -0.543052137, -0.525114477, 0.1...   \n",
      "\n",
      "                                         resume_bert  \\\n",
      "0  [-0.933770418, -0.155764461, -0.401959687, 0.0...   \n",
      "1  [-0.912764013, 0.236520529, -0.479016751, -0.0...   \n",
      "2  [-0.477572799, -0.344656736, -0.686233759, 0.1...   \n",
      "3  [-1.15020216, -0.297064811, -0.456645876, -0.4...   \n",
      "4  [-1.1292547, -0.10180103, -0.269980609, -0.380...   \n",
      "\n",
      "                            job_description_word2vec  \\\n",
      "0  [-0.4533605, 0.37356848, -0.00404537, 0.111069...   \n",
      "1  [-0.39174914, 0.55327415, -0.06747422, 0.07611...   \n",
      "2  [-0.38531312, 0.5408037, -0.00435989, 0.018309...   \n",
      "3  [-0.41355154, 0.508325, 0.01384492, 0.05911843...   \n",
      "4  [-0.38863534, 0.5051077, -0.07876955, 0.071466...   \n",
      "\n",
      "                                 transcript_word2vec  \\\n",
      "0  [-0.42832482, 0.54906446, 0.00907241, 0.068270...   \n",
      "1  [-0.36496997, 0.6715688, 0.07627022, 0.0206103...   \n",
      "2  [-0.37096733, 0.63587874, 0.06612763, 0.017948...   \n",
      "3  [-0.40843606, 0.514646, -0.00260729, 0.0677779...   \n",
      "4  [-0.40718457, 0.5918017, 0.03645066, 0.0525984...   \n",
      "\n",
      "                                     resume_word2vec  \n",
      "0  [-0.496016473, 0.443078548, -0.107665651, 0.12...  \n",
      "1  [-0.516340852, 0.364454627, -0.146326035, 0.17...  \n",
      "2  [-0.4791812, 0.40962663, -0.06950963, 0.129839...  \n",
      "3  [-0.45877597, 0.4127093, -0.09734197, 0.107441...  \n",
      "4  [-0.53417623, 0.358433247, -0.166723281, 0.170...  \n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# Load the CSV file\n",
    "df = pd.read_excel('B:\\OneDrive - Amity University\\Desktop\\Intern\\Infosys\\Prediction\\processed_dataset_with_embeddings.xlsx')\n",
    "\n",
    "# Example: Assuming the embeddings are in columns 6, 7, and 8 (adjust according to your actual column names)\n",
    "embedding_columns = ['job_description_bert', 'transcript_bert', 'resume_bert','job_description_word2vec','transcript_word2vec','resume_word2vec']  # replace with your actual column names\n",
    "\n",
    "# Function to convert the string of numbers into a list of floats\n",
    "def convert_to_float(embedding_str):\n",
    "    embedding_list = embedding_str.strip('[]').split()  # remove the brackets and split the numbers\n",
    "    return [float(num) for num in embedding_list]  # convert each number to float\n",
    "\n",
    "# Apply the function to the relevant columns\n",
    "for col in embedding_columns:\n",
    "    df[col] = df[col].apply(convert_to_float)\n",
    "\n",
    "# Optionally, check the first few rows of the DataFrame to confirm\n",
    "print(df.head())\n",
    "\n",
    "# Save the modified DataFrame to a new CSV if needed\n",
    "df.to_csv('modified_embeddings_bert_word2vec.csv', index=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "XGBoost Model Predictions: [1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 0 0 0 1 1 0 1 0 1 1 1 1 1 1 1 1\n",
      " 1 1 1 1 1 0 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 0 1 1 1 1 1 1\n",
      " 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 0 1 1 1 1 0 0 1 1]\n",
      "Predictions saved to 'predictions.csv'.\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import xgboost as xgb\n",
    "import ast\n",
    "\n",
    "def process_embedding_string(embedding_str):\n",
    "    \"\"\"Convert string representation of embeddings to numpy array\"\"\"\n",
    "    try:\n",
    "        # Convert string representation of list to actual list\n",
    "        return np.array(ast.literal_eval(embedding_str))\n",
    "    except:\n",
    "        return None\n",
    "\n",
    "# Load BERT-Word2Vec embeddings CSV file (prediction data)\n",
    "bert_word2vec_df = pd.read_csv(r'B:\\OneDrive - Amity University\\Desktop\\Intern\\Infosys\\Prediction\\modified_embeddings_bert_word2vec.csv')\n",
    "\n",
    "# Process embedding columns (same as training)\n",
    "embedding_columns_bert = ['job_description_bert', 'resume_bert', 'transcript_bert', 'job_description_word2vec', 'resume_word2vec', 'transcript_word2vec']\n",
    "\n",
    "# Apply the process_embedding_string function to each embedding column\n",
    "for col in embedding_columns_bert:\n",
    "    bert_word2vec_df[col] = bert_word2vec_df[col].apply(process_embedding_string)\n",
    "\n",
    "# Create feature matrix (only BERT-Word2Vec embeddings)\n",
    "def create_feature_matrix(df, embedding_columns):\n",
    "    features = []\n",
    "    for _, row in df.iterrows():\n",
    "        embeddings = []\n",
    "        for col in embedding_columns:\n",
    "            embedding = row[col]\n",
    "            if embedding is not None:  # Check for None values after conversion\n",
    "                embeddings.append(embedding)\n",
    "        \n",
    "        if embeddings:  # Only concatenate if embeddings are available\n",
    "            combined_embedding = np.concatenate(embeddings)\n",
    "            features.append(combined_embedding)\n",
    "    \n",
    "    return np.vstack(features) if features else None  # Handle the case where no valid features are created\n",
    "\n",
    "# Create feature matrix for prediction data\n",
    "X_bert_word2vec = create_feature_matrix(bert_word2vec_df, embedding_columns_bert)\n",
    "\n",
    "# Load the saved XGBoost model\n",
    "xgb_model = xgb.XGBClassifier()\n",
    "xgb_model.load_model(r'B:\\OneDrive - Amity University\\Desktop\\Intern\\Infosys\\Prediction\\complex_xgb_model1.json')\n",
    "\n",
    "# Predict using XGBoost model\n",
    "y_pred_xgb = xgb_model.predict(X_bert_word2vec)\n",
    "print(f'XGBoost Model Predictions: {y_pred_xgb}')\n",
    "\n",
    "# Add predictions to the DataFrame\n",
    "bert_word2vec_df['xgb_predictions'] = y_pred_xgb\n",
    "\n",
    "# Save predictions to a new CSV file if needed\n",
    "bert_word2vec_df.to_csv(r'predictions.csv', index=False)\n",
    "print(\"Predictions saved to 'predictions.csv'.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Email sent successfully!\n"
     ]
    }
   ],
   "source": [
    "import smtplib\n",
    "from email.mime.multipart import MIMEMultipart\n",
    "from email.mime.text import MIMEText\n",
    "from email.mime.base import MIMEBase\n",
    "from email import encoders\n",
    "import os\n",
    "\n",
    "class EmailSender:\n",
    "    def __init__(self, provider='gmail'):\n",
    "        self.providers = {\n",
    "            'gmail': {\n",
    "                'smtp_server': 'smtp.gmail.com',\n",
    "                'port': 587\n",
    "            },\n",
    "            'outlook': {\n",
    "                'smtp_server': 'smtp-mail.outlook.com',\n",
    "                'port': 587\n",
    "            },\n",
    "            'yahoo': {\n",
    "                'smtp_server': 'smtp.mail.yahoo.com',\n",
    "                'port': 587\n",
    "            }\n",
    "        }\n",
    "        self.provider = provider\n",
    "\n",
    "    def send_email(self, sender_email, sender_password, to_email, subject, body, file_path=None):\n",
    "        try:\n",
    "            # Configure SMTP server details\n",
    "            smtp_server = self.providers[self.provider]['smtp_server']\n",
    "            port = self.providers[self.provider]['port']\n",
    "\n",
    "            # Create message\n",
    "            message = MIMEMultipart()\n",
    "            message[\"From\"] = sender_email\n",
    "            message[\"To\"] = to_email\n",
    "            message[\"Subject\"] = subject\n",
    "            message.attach(MIMEText(body, \"plain\"))\n",
    "\n",
    "            # Attach file if provided\n",
    "            if file_path and os.path.exists(file_path):\n",
    "                with open(file_path, \"rb\") as attachment:\n",
    "                    part = MIMEBase(\"application\", \"octet-stream\")\n",
    "                    part.set_payload(attachment.read())\n",
    "                \n",
    "                encoders.encode_base64(part)\n",
    "                part.add_header(\n",
    "                    \"Content-Disposition\",\n",
    "                    f\"attachment; filename={os.path.basename(file_path)}\",\n",
    "                )\n",
    "                message.attach(part)\n",
    "\n",
    "            # Send email\n",
    "            with smtplib.SMTP(smtp_server, port) as server:\n",
    "                server.starttls()\n",
    "                server.login(sender_email, sender_password)\n",
    "                server.send_message(message)\n",
    "                print(\"Email sent successfully!\")\n",
    "                return True\n",
    "\n",
    "        except Exception as e:\n",
    "            print(f\"Email sending failed: {e}\")\n",
    "            return False\n",
    "\n",
    "# Example usage\n",
    "if __name__ == \"__main__\":\n",
    "    email_sender = EmailSender(provider='gmail')  # Can change to 'outlook' or 'yahoo'\n",
    "    email_sender.send_email(\n",
    "        sender_email=\"akash1198770@gmail.com\",\n",
    "        sender_password=\"ndsy rina axhj yrgq\",\n",
    "        to_email=\"siddharthsharma0956@gmail.com\",\n",
    "        subject=\"Test Email\",\n",
    "        body=\"This is a test email.\",\n",
    "        file_path=\"B:\\OneDrive - Amity University\\Desktop\\Intern\\Infosys\\Prediction\\predictions.csv\"\n",
    "    )"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
