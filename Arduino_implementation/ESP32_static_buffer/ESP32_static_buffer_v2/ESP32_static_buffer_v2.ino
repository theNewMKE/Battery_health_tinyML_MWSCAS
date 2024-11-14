/* Edge Impulse ingestion SDK
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

/* Includes ---------------------------------------------------------------- */
#include <bat_tinyML_conf_inferencing.h>

// static const float features[] = {
//     // copy raw features here (for example from the 'Live classification' page)
//     // see https://docs.edgeimpulse.com/docs/running-your-impulse-arduino
// };

// define the biffer size for each input
const int BUFFER_SIZE = 160;
// create a features array
// this will be copied to out_ptr in raw_feature_get_data function
static float features[BUFFER_SIZE]; 
// set the data received flag as false
bool data_received_flag = false;
// inference count number
int inference_count=0;

/**
 * @brief      Copy raw feature data in out_ptr
 *             Function called by inference library
 *
 * @param[in]  offset   The offset
 * @param[in]  length   The length
 * @param      out_ptr  The out pointer
 *
 * @return     0
 */
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}

void print_inference_result(ei_impulse_result_t result);

/**
 * @brief      Arduino setup function
 */
void setup()
{
    // put your setup code here, to run once:
    Serial.begin(9600);
    // comment out the below line to cancel the wait for USB connection (needed for native USB)
    // while (!Serial);
    // Serial.println("Edge Impulse Inferencing Demo");
}

/**
 * @brief      Arduino main function
 */
void loop()
{
    // ei_printf("Edge Impulse standalone inferencing (Arduino)\n");


    // int8_t x_quantized = 0;
    // int8_t y_quantized = 0;
    // float y_pred = 0.0;

    // (1) Input Data
    // static buffer test for one data point
    // this should be commented out when running the actual program
    // static float data_buffer[] = { 
    //   0.85769287, 0.91666667, 0.70801953, 0.15812045, 0.05682766, 0.15136703,  0.05509788, 0.27653167, 0.90251432, 0.05562347, 0.72383908, 0.65864495, 0.2040728,  0.67086314, 0.2138099,  0.2569055,  0.66249458, 0.45141513, 0.90088181, 0.7110021,  0.04635209, 0.7217374,  0.04984498, 0.22107996, 0.13310408, 1.,         0.78204594, 0.23074787, 0.,         0.21222381, 0.,         0.70404984, 0.17764025, 1.,         0.93062862, 0.58160558, 0.,         0.56453664, 0.,         0.54257529, 0.49988083, 0.5, 0.55323014, 0.4087375,  0.34127499, 0.40996662, 0.34037559, 0.27455867, 0.71061486, 0.,         0.64553609, 0.71566087, 0.3525231,  0.72646638, 0.36442555, 0.18442368, 0.19252182, 1.,         0.784733,   0.21223621, 0.,         0.19578763, 0.,         0.48535826, 0.00955669, 0.33348151, 0.24603565, 0.99851907, 0.97305398, 0.99818789, 0.98198246, 0.57455867, 0.99750436, 0.33348151, 0.60031034, 0.59940763, 0.46327369, 0.61478302, 0.48357693, 0.11557632, 0.53459876, 0.33348151, 0.46599553, 0.74639023, 0.725163,   0.75593705, 0.74782532, 0.3549325,  0.12364497, 0., 0.49294176, 0.91484635, 0.7227836,  0.91654745, 0.73476836, 0.47715472, 0.99523324, 0.33353708, 0.65923627, 0.58867086, 0.36917895, 0.60209824, 0.39189476, 0.21079958, 0.00360049, 1.,         0.78272717, 0.27249167, 0.,         0.25116834, 0.,         0.41214953, 0.54513806, 1., 0.76535594, 0.07361471, 0.,         0.06698458, 0.,         0.33509865, 0.81480571, 0.74998148, 0.63721001, 0.24302727, 0.1663422,  0.23977905, 0.16305253, 0.2290758,  0.58123993, 0.58337038, 0.58331756, 0.34656917, 0.27734001, 0.34721825, 0.27203472, 0.28483904, 0.72923879, 0.75, 0.62245014, 0.24737751, 0.17397485, 0.24393578, 0.17010364, 0.17237799, 0.45595458, 1.,         0.90425009, 0.41537085, 0.,         0.40255921, 0.,         0.2384216,  0.19286309, 1.,         0.87109715, 0.43382081, 0.,         0.41383723, 0.,         0.59252336
    // };

    // here, we received 20*8 float numbers from Python program running on PC;
    float received_float = 0.0;
    int received_count = 0; // set this to 160 when using fxed data buffer
    data_received_flag = false; // set this to true when using fixed data buffer
    float data_buffer[BUFFER_SIZE];

    while (Serial.available() && received_count < BUFFER_SIZE){
      data_received_flag = true;
      received_float = Serial.parseFloat();  //parse received received string from PC to float numbers
      // store the byte to data_buffer array
      data_buffer[received_count] = received_float;

      // Serial.print(received_count);
      received_count++;
    }

    // if we received a datapoint with exactly 20*8=160 values, the
    // we pass taht to the moel to make a prediction;
    if (data_received_flag == true){
      // this will be used inside Python program
      Serial.print(received_count);
      Serial.print("\r\n");

      if (received_count == BUFFER_SIZE){
        // to calculate execute time of inference, we record here
        // start time of inference 
        // long int t1_start = millis();

        // copy the data_buffer into the features array
        // the features array is defined at the begining
        memcpy(features, data_buffer, BUFFER_SIZE*sizeof(float));

    // size check by EI
    if (sizeof(features) / sizeof(float) != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
        ei_printf("The size of your 'features' array is not correct. Expected %lu items, but had %lu\n",
            EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, sizeof(features) / sizeof(float));
        // delay(1000);
        return;
    }

    ei_impulse_result_t result = { 0 };

    // the features are stored into flash, and we don't want to load everything into RAM
    signal_t features_signal;
    features_signal.total_length = sizeof(features) / sizeof(features[0]);
    features_signal.get_data = &raw_feature_get_data;

    // invoke the impulse
    EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false /* debug */);
    if (res != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", res);
        return;
    }

        // end time of inference
        // long int t2_end = millis();

        // Serial.print("Inference time: ");
        // Serial.print(t2_end-t1_start);
        // Serial.print("\r\n");

        // print inference return code
        // ei_printf("run_classifier returned: %d\r\n", res);
        // print_inference_result(result);

        // print the result
        // the result.classification is taken from print_inference_result function
        Serial.print(result.classification[0].value);

        // increment the inference_counter;
        inference_count += 1;
        if (inference_count >= 10000) {
          inference_count = 0;
        }        
      }
    }
    delay(50);
}

void print_inference_result(ei_impulse_result_t result) {

    // Print how long it took to perform inference
    ei_printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\r\n",
            result.timing.dsp,
            result.timing.classification,
            result.timing.anomaly);

    // Print the prediction results (object detection)
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    ei_printf("Object detection bounding boxes:\r\n");
    for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
        if (bb.value == 0) {
            continue;
        }
        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                bb.label,
                bb.value,
                bb.x,
                bb.y,
                bb.width,
                bb.height);
    }

    // Print the prediction results (classification)
#else
    ei_printf("Predictions:\r\n");
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        ei_printf("  %s: ", ei_classifier_inferencing_categories[i]);
        ei_printf("%.5f\r\n", result.classification[i].value);
    }
#endif

    // Print anomaly result (if it exists)
#if EI_CLASSIFIER_HAS_ANOMALY
    ei_printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif

#if EI_CLASSIFIER_HAS_VISUAL_ANOMALY
    ei_printf("Visual anomalies:\r\n");
    for (uint32_t i = 0; i < result.visual_ad_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.visual_ad_grid_cells[i];
        if (bb.value == 0) {
            continue;
        }
        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                bb.label,
                bb.value,
                bb.x,
                bb.y,
                bb.width,
                bb.height);
    }
#endif

}