If you get an error like this for the first time uploading:
/home/feiticeir0/Documents/Arduino/libraries/XIAO_Lamps_inferencing/src/edge-impulse-sdk/tensorflow/lite/micro/kernels/conv.cpp:1789:67: error: either all initializer clauses should be designated or none of them should be
 1789 |                                 .channels = input->dims->data[3], 1


check out this website:
https://forum.edgeimpulse.com/t/error-compiling-arduino-library-for-xiao-esp32s3-sense/8901

To be more straightforward:

Hi all.
I’ve solved the problem by using the solution from here:
https://www.hackster.io/mjrobot/tinyml-made-easy-image-classification-cb42ae#toc-connecting-sense-module–expansion-board-2 109

Just editing the file ei_classifier_config.h in exported Arduino library folder: /scr/edge-impulse-sdk/classifier/:

Disabling #define EI_CLASSIFIER_TFLITE_ENABLE_ESP_NN 1 and set it to 0 will allow the project to compile .

Thank you