# Diabetes Patients Readmissions Prediction

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

