# Group 11 Week 3 

For this week's submission, we developed the individual model pipeline experiments for the proposed University Communications Intelligence system. The system processes university notices and converts unstructured communication into structured and audience-specific information. The current pipeline consists of four major components: Named Entity Recognition (NER), notice classification, audience classification, and persona-based summarization.

## 1. Named Entity Recognition and Information Extraction

The NER module identifies important information from university communications using **spaCy** together with custom extraction rules. It extracts entities such as **dates, times, organizations, people, email addresses, URLs, target audiences, and required actions**.

In addition to individual entities, the system identifies relationships between extracted information. For example, an action can be associated with its intended audience and deadline. This allows a notice to be represented in a more structured form rather than being treated only as plain text.

## 2. Notice Category Classification

The category classifier determines the primary type of a university notice. Six categories are currently considered:

* Academic
* Admission
* Announcements
* Examination
* General
* Hostel

The notice text is converted into numerical features using **TF-IDF with unigram and bigram features**. These features are then passed to a **Logistic Regression classifier** with balanced class weights. A **LinearSVC** model is additionally evaluated as a baseline on the validation set.

## 3. Audience Classification

The audience classifier identifies the groups for whom a notice is relevant. It uses **multi-label classification**, allowing a single notice to be associated with more than one audience.

The current audience classes are:

* Faculty
* Students

TF-IDF features are passed to a **One-vs-Rest Logistic Regression** model. This approach trains an independent classifier for each audience, allowing multiple audiences to be predicted simultaneously. A **LinearSVC** model is also used as a comparison baseline.

As a further improvement, the **Administrator** audience class will be incorporated into the classifier so that notices relevant to administrators can also be identified.

## 4. Persona-Based Summarization

After identifying the relevant audience, the system generates a concise summary tailored to that audience. A **BART-based sequence-to-sequence model** is used for summarization.

The audience is included in the summarization, allowing the same notice to be summarized differently depending on the intended audience. For example, information relevant to students can be emphasized differently from information intended for faculty or administrators.

The generated summaries are evaluated using **ROUGE-1, ROUGE-2, and ROUGE-L**. In addition, a factual consistency check compares dates and times in the generated summary with those present in the original notice to identify potential inconsistencies.

The final output of this stage is therefore an **audience-specific, concise summary of the original university notice**, which can be presented to the identified audience along with the structured information extracted by the earlier stages.

## 5. Data Integrity and Evaluation

The classification models use a controlled **train-validation-test split**. The training set contains **108 documents**, including real and synthetic training examples, while the validation and test sets contain only real documents. The test set remains untouched during model development.

The pipeline also performs explicit checks to ensure that documents do not overlap between the different splits and that synthetic documents are restricted to the training set. This helps prevent data leakage and provides a more reliable evaluation of the developed models.

Overall, these experiments establish the individual components of the proposed system and provide the foundation for integrating them into a single end-to-end **University Communications Intelligence pipeline**.


## 6. Improvements and Future Work

The current experiments establish the core individual components of the proposed system. The following improvements will be considered during further development:

* **Expand audience classification** by adding the **Administrator** class alongside Faculty and Students.
* **Improve classification performance** by comparing the current TF-IDF-based models with alternative feature representations and model configurations.
* **Improve summarization quality** by further fine-tuning the BART model on the available university communication data.
* **Strengthen factual consistency** by extending the current date and time checks to other important information extracted by the NER module.
* **Integrate the individual components** into a single end-to-end pipeline, where extracted entities and classification results are used to produce the final audience-specific communication.
* **Evaluate the complete system** on unseen university notices to assess its effectiveness under realistic usage conditions.
