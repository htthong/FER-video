# Video Emotion Recognition

## Running
Create folders to save log and checkpoint
~~~~
$ !mkdir log
$ !mkdir checkpoint
~~~~

For training stage, run code below
~~~~
$ python main.py --txt_train TEXT_TRAIN_FILE_PATH --txt_test TEXT_TEST_FILE_PATH --root_data DATA_FOLDER 
~~~~
Example:
~~~~
$ python main.py --txt_train ./FERV39K_train.txt --txt_test ./FERV39K_test.txt --root_data ../data/FERV39K/2_ClipsforFaceCrop 
~~~~
For evaluate stage, comment main function and uncomment test function

For run FER-app
~~~~
$ python manage.py runserver 
~~~~
## References

We referenced the repo of [Former-DFER](https://github.com/zengqunzhao/Former-DFER/) for the code.
