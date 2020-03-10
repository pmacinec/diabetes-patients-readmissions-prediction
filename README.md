# Diabetes Patients Early Readmissions Prediction

**Authors:** Peter Macinec, Frantisek Sefcik

## Installation and running

To run this project, please make sure you have Docker installed. After, follow the steps:
1. Get into project root repository.
1. Build docker image:
    ```
    docker build -t dprp_image .
    ```
1. Run docker container using command: 
    ```
    docker run -it --name dprp_con --rm -p 8888:8888 -v $(pwd):/project/ dprp_image
    ```

Do not forget to add dataset into `data` folder. Remember that csv file must be named `data.csv`.


## Dataset

Dataset we used in this project is well-known *Diabetes 130-US hospitals for years 1999-2008 Data Set*.

**Note:** Following description taken from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008).

The dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes. Information was extracted from the database for encounters that satisfied the following criteria:

1. It is an inpatient encounter (a hospital admission).
2. It is a diabetic encounter, that is, one during which any kind of diabetes was entered to the system as a diagnosis.
3. The length of stay was at least 1 day and at most 14 days.
4. Laboratory tests were performed during the encounter.
5. Medications were administered during the encounter.

The data contains such attributes as patient number, race, gender, age, admission type, time in hospital, medical specialty of admitting physician, number of lab test performed, HbA1c test result, diagnosis, number of medication, diabetic medications, number of outpatient, inpatient, and emergency visits in the year before the hospitalization, etc.

Links on dataset:
* [Kaggle - Diabetes 130 US hospitals for years 1999-2008](https://www.kaggle.com/brandao/diabetes)
* [Original paper with data](https://www.hindawi.com/journals/bmri/2014/781670/) - go to Supplementary Materials section 
* [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)
