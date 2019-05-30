# Training.

Here you will find all training scripts. 

Things to take into account:
- Please make sure that the instructions for training reside in ```runtraining.sh```. This is the entrypoint.
- Please make sure that training artifacts end up in ```/whhdata/models```.
- Please make sure that training artifacts always have a date and a time in the filename.
- Please make sure that each training uses TensorBoard. TensorBoard logs should end up in ```/whhdata/models/logs```.