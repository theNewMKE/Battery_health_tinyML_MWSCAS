// Description:
// receive 20*8 datapoint from python program running on PC
// use that as input into tynyML model, whcih makes prediction
// send prediction back to PC
//
// this Arduino sketch is developed starting from the Arduino examples
// "hello_world" and "micro_speech" from the tinyML book of Pete Warden.
// the latest version of these examples is available here:
// https://github.com/tensorflow/tflite-micro-arduino-examples/tree/main
// the (development, training, etc.) files are here:
// https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples

#include <TensorFlowLite.h>
// #include <TensorFlowLite_ESP32.h>
#include "main_functions.h"
#include "model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"


// (1) Declare Variables
// Globals, used for compatibility with Arduino-style sketches.
// This is an "unnanmed namespace" - allows us to create identifiers
// that are unique wihin a file; they are also called "anonymous namespaces";
// these identifiers are known only within this file;
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 80 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
int8_t* model_input_buffer = nullptr;

// float_t* model_input_buffer = nullptr;
} // namespace


#define BUFFER_SIZE 160 // should be exactly the same as "kFeatureElementCount"
float data_buffer[BUFFER_SIZE];
bool data_received_flag = false;
// a fixed 160 datapoint buffer; used for local debug;
// X_test_reshaped index 0 (+2 in excel)
// float data_buffer_fixed[] = {0.85769287, 0.91666667, 0.70801953, 0.15812045, 0.05682766, 0.15136703,  0.05509788, 0.27653167, 0.90251432, 0.05562347, 0.72383908, 0.65864495, 0.2040728,  0.67086314, 0.2138099,  0.2569055,  0.66249458, 0.45141513, 0.90088181, 0.7110021,  0.04635209, 0.7217374,  0.04984498, 0.22107996, 0.13310408, 1.,         0.78204594, 0.23074787, 0.,         0.21222381, 0.,         0.70404984, 0.17764025, 1.,         0.93062862, 0.58160558, 0.,         0.56453664, 0.,         0.54257529, 0.49988083, 0.5, 0.55323014, 0.4087375,  0.34127499, 0.40996662, 0.34037559, 0.27455867, 0.71061486, 0.,         0.64553609, 0.71566087, 0.3525231,  0.72646638, 0.36442555, 0.18442368, 0.19252182, 1.,         0.784733,   0.21223621, 0.,         0.19578763, 0.,         0.48535826, 0.00955669, 0.33348151, 0.24603565, 0.99851907, 0.97305398, 0.99818789, 0.98198246, 0.57455867, 0.99750436, 0.33348151, 0.60031034, 0.59940763, 0.46327369, 0.61478302, 0.48357693, 0.11557632, 0.53459876, 0.33348151, 0.46599553, 0.74639023, 0.725163,   0.75593705, 0.74782532, 0.3549325,  0.12364497, 0., 0.49294176, 0.91484635, 0.7227836,  0.91654745, 0.73476836, 0.47715472, 0.99523324, 0.33353708, 0.65923627, 0.58867086, 0.36917895, 0.60209824, 0.39189476, 0.21079958, 0.00360049, 1.,         0.78272717, 0.27249167, 0.,         0.25116834, 0.,         0.41214953, 0.54513806, 1., 0.76535594, 0.07361471, 0.,         0.06698458, 0.,         0.33509865, 0.81480571, 0.74998148, 0.63721001, 0.24302727, 0.1663422,  0.23977905, 0.16305253, 0.2290758,  0.58123993, 0.58337038, 0.58331756, 0.34656917, 0.27734001, 0.34721825, 0.27203472, 0.28483904, 0.72923879, 0.75, 0.62245014, 0.24737751, 0.17397485, 0.24393578, 0.17010364, 0.17237799, 0.45595458, 1.,         0.90425009, 0.41537085, 0.,         0.40255921, 0.,         0.2384216,  0.19286309, 1.,         0.87109715, 0.43382081, 0.,         0.41383723, 0.,         0.59252336};
// X_test_reshaped index 46 (+2 in excel)
// float data_buffer_fixed[] = {0.2392082929696497,  0.8333333333333335,0.641751504371192, 0.22541034184869796,0.13853094774574334,0.21797806390081073,0.13750553636283105,0.25950155763239735, 0.754687133134211, 0.0, 0.645119782008099, 0.7086264346538318, 0.3525230987917555, 0.7195040534096327, 0.36465585968642034, 0.24859813084112403, 0.4442026421514563, 1.0, 0.8531582333573023, 0.3045476983833148, 0.0, 0.2901525989508822, 0.0, 0.29823468328141445, 0.43584124635995514, 0.0, 0.6772508799152256, 0.7660125879303961, 0.3525230987917555, 0.7752026704816405, 0.3701390734343165, 0.30737279335410506, 0.8968186543215949, 0.9012558346299179, 0.9972372554214132, 0.5610576329754412, 0.0, 0.5648466062629153, 0.0, 0.2569055036344743, 0.05088029099968058, 1.0, 0.8280664572531508, 0.39689004072565714, 0.0, 0.37165792401843906, 0.0, 0.4844236760124616, 0.31648117492023903, 0.8424279469511745, 0.9996972334708399, 0.7432123904726643, 0.0, 0.7462565569861708, 0.0, 0.3492211838006227, 0.8066627975067633, 0.7961213603022895, 0.9962154183854975, 0.6655251141552512, 0.0, 0.673931012557622, 0.0, 0.28753894080997, 0.6914915919600122, 0.9594169074609173, 0.9833478408961894, 0.5571393311119338, 0.0, 0.5555078683834049, 0.0, 0.21464174454828822, 0.43869472071260296, 1.0, 0.9285092533020475, 0.4707824262618783, 0.0, 0.45966460022254013, 0.0, 0.21204569055036515, 0.5349325561777246, 0.9999814773653406, 0.7772395261703818, 0.10992842157225718, 0.0, 0.10116833571769193, 0.0, 0.2887850467289752, 0.3554528308730374, 0.7500000000000001, 0.6147674374597888, 0.26616685178329014, 0.1925774852445845, 0.2616992529009697, 0.1909115067765081, 0.3343717549325049, 0.4116605101230635, 0.932151589242054, 0.9923172993225599, 0.6407503393804763, 0.0, 0.6392306469559689, 0.0, 0.2867082035306332, 0.8043091897108688, 1.0, 0.920069636301707, 0.39689004072565714, 0.0, 0.38739469082816724, 0.0, 0.36812045690550477, 0.03554076117293739, 0.31857079350966894, 0.8746546569276769, 0.9645810193755399, 0.08652390222799047, 0.9636464790971229, 0.09402072814243954, 0.5420560747663572, 0.8146543187385249, 0.7499814773653406, 0.637210006433789, 0.24302727384919162, 0.1663422020333117, 0.2397790494357018, 0.1630525290105412, 0.2290758047767376, 0.00018396048845025783, 1.0, 0.7488551640616129, 0.11612982845859558, 0.0, 0.10732792878715626, 0.0, 0.4186915887850482, 0.19399068535770533, 0.0, 0.583090489346403, 0.8722695298037763, 0.6124347208059082, 0.8768717215069147, 0.6299317920099211, 0.5169262720664598, 0.0, 1.0, 0.7205086477689894, 0.042144884610638034, 0.0, 0.038396121443331745, 0.0, 0.46490134994807875, 0.9963057807569328, 0.9999814773653406, 0.9846724444612648, 0.4246575342465753, 0.0, 0.4258146558575744, 0.0, 0.2828660436137085};
// X_test_reshaped index 47 (+2 in excel)
float data_buffer_fixed[] = {0.754687133134211,  0.0, 0.645119782008099, 0.7086264346538318, 0.3525230987917555, 0.7195040534096327, 0.36465585968642034, 0.24859813084112403, 0.4442026421514563, 1.0, 0.8531582333573023, 0.3045476983833148, 0.0, 0.2901525989508822, 0.0, 0.29823468328141445, 0.43584124635995514, 0.0, 0.6772508799152256, 0.7660125879303961, 0.3525230987917555, 0.7752026704816405, 0.3701390734343165, 0.30737279335410506, 0.8968186543215949, 0.9012558346299179, 0.9972372554214132, 0.5610576329754412, 0.0, 0.5648466062629153, 0.0, 0.2569055036344743, 0.05088029099968058, 1.0, 0.8280664572531508, 0.39689004072565714, 0.0, 0.37165792401843906, 0.0, 0.4844236760124616, 0.31648117492023903, 0.8424279469511745, 0.9996972334708399, 0.7432123904726643, 0.0, 0.7462565569861708, 0.0, 0.3492211838006227, 0.8066627975067633, 0.7961213603022895, 0.9962154183854975, 0.6655251141552512, 0.0, 0.673931012557622, 0.0, 0.28753894080997, 0.6914915919600122, 0.9594169074609173, 0.9833478408961894, 0.5571393311119338, 0.0, 0.5555078683834049, 0.0, 0.21464174454828822, 0.43869472071260296, 1.0, 0.9285092533020475, 0.4707824262618783, 0.0, 0.45966460022254013, 0.0, 0.21204569055036515, 0.5349325561777246, 0.9999814773653406, 0.7772395261703818, 0.10992842157225718, 0.0, 0.10116833571769193, 0.0, 0.2887850467289752, 0.3554528308730374, 0.7500000000000001, 0.6147674374597888, 0.26616685178329014, 0.1925774852445845, 0.2616992529009697, 0.1909115067765081, 0.3343717549325049, 0.4116605101230635, 0.932151589242054, 0.9923172993225599, 0.6407503393804763, 0.0, 0.6392306469559689, 0.0, 0.2867082035306332, 0.8043091897108688, 1.0, 0.920069636301707, 0.39689004072565714, 0.0, 0.38739469082816724, 0.0, 0.36812045690550477, 0.03554076117293739, 0.31857079350966894, 0.8746546569276769, 0.9645810193755399, 0.08652390222799047, 0.9636464790971229, 0.09402072814243954, 0.5420560747663572, 0.8146543187385249, 0.7499814773653406, 0.637210006433789, 0.24302727384919162, 0.1663422020333117, 0.2397790494357018, 0.1630525290105412, 0.2290758047767376, 0.00018396048845025783, 1.0, 0.7488551640616129, 0.11612982845859558, 0.0, 0.10732792878715626, 0.0, 0.4186915887850482, 0.19399068535770533, 0.0, 0.583090489346403, 0.8722695298037763, 0.6124347208059082, 0.8768717215069147, 0.6299317920099211, 0.5169262720664598, 0, 1.0, 0.7205086477689894, 0.042144884610638034, 0.0, 0.038396121443331745, 0.0, 0.46490134994807875, 0.9963057807569328, 0.9999814773653406, 0.9846724444612648, 0.4246575342465753, 0.0, 0.4258146558575744, 0.0, 0.2828660436137085, 0.5335156108651388, 0.7153256279173151, 0.9809635544790526, 0.743736887572504, 0.0, 0.7531235097758703, 0.0, 0.31433021806853745};
// X_test_reshaped index 114 (+2 in excel)
// float data_buffer_fixed[] ={0.6254926441017256,  0.0,  0.6402755175415357, 0.7226952980377638, 0.35249219739810267, 0.7342870767763473, 0.36279564177517937, 0.22076843198338736, 0.9306773133052801, 0.9999629547306811, 0.9273360330015518, 0.3599592743428359, 0.0, 0.3529089175011922, 0.0, 0.1855659397715499, 0.3293694232770732, 0.0, 0.565340801574386, 0.7985931136616068, 0.5753221470288309, 0.8069623271340011, 0.5884666489503055, 0.2596053997923171, 0.2672033203051937, 0.8904386159887384, 0.9992052378609545, 0.7352523756633345, 0.0, 0.7344857733269751, 0.0, 0.23167185877466423, 0.12175907597771383, 0.9999814773653406, 0.813079514059721, 0.3230285079600148, 0.0, 0.301239866475918, 0.0, 0.5309449636552444, 0.7775614758065973, 0.8499296139882938, 0.9996215418385498, 0.6374799456991237, 0.0, 0.6428469241773963, 0.0, 0.17310488058151563, 0.04433075449207559, 0.8509483588945693, 0.9997729251031301, 0.8988029125015425, 0.0, 0.8930456207280242, 0.0, 0.5388369678089298, 0.32831728243871516, 0.9960546788175152, 0.9643113953752415, 0.5787054177465136, 0.0, 0.5701557780956923, 0.0, 0.403426791277262, 0.9639078411643586, 0.849799955545677, 0.9995836960224048, 0.5485931136616068, 0.0, 0.5560960101732634, 0.0, 0.15192107995846627, 0.1189068170370494, 0.0, 0.5769216213147635, 0.9170677526841909, 0.6495781959766386, 0.9188364329995232, 0.6675790592612276, 0.473416407061265, 0.7951943071668711, 1.0, 0.8306399727510124, 0.21223620881155122, 0.0, 0.2005563503417581, 0.0, 0.328556593977158, 0.7820869902595469, 0.0, 0.5730235022518261, 0.7038134024435394, 0.46395352430394604, 0.714544587505961, 0.4712197714589423, 0.2517133956386317, 0.07673818532147209, 0.9730866118396682, 0.9805472505014573, 0.760366530914476, 0.0, 0.7463519313304722, 0.0, 0.5251298026998974, 0.8849190194912177, 0.6538490034822555, 0.963403095787761, 0.6487103541898062, 0.009270418095856124, 0.6588300747099031, 0.01011604216493932, 0.29657320872274084, 0.6557472528886858, 1.0, 0.7902963327404157, 0.1383438232753301, 0.0, 0.12848513749801305, 0.0, 0.2656282450674965, 0.6022924769693454, 1.0, 0.9130681603148777, 0.39689004072565714, 0.0, 0.38689397552058497, 0.0, 0.29231568016614773, 0.7988364552909974, 1.0, 0.8482004314423041, 0.24913612242379363, 0.0, 0.23719599427753937, 0.0, 0.2499480789200419, 0.44454390980189923, 1.0, 0.954774249706695, 0.5250833024805627, 0.0, 0.5166428230805914, 0.0, 0.3334371754932519, 0.43822219622305486, 1.0, 0.8289747568406315, 0.24913612242379363, 0.0, 0.23516928946113497, 0.0, 0.24402907580477518, 0.0008148155181934198, 1.0, 0.9024334859781252, 0.560008638775762, 0.0, 0.5425846447305676, 0.0, 0.467808930425754};
///////////////////////////////////////////////////////////////////////////////
//
// setup
//
///////////////////////////////////////////////////////////////////////////////

void setup() 
{
  // we will use Serial to talk to Python program on PC;
  // to receive datpoints of 20*8 float numbers and to send back prediction;
  Serial.begin(9600);
  while(!Serial);

  // init;
  tflite::InitializeTarget();

  // (2) Load Model
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // (3) Resolve Operators
  // This pulls in all the operation implementations (Ops) we need.
  static tflite::AllOpsResolver resolver;

  // (4) Initialize Interpreter
  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  // TODO: look more into:
  // https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter
  // https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Interpreter
  // https://groups.google.com/a/tensorflow.org/g/micro/c/w38Q81rOR-w
  // https://coral.ai/docs/reference/micro/tensorflow/#tflm-interpreter

  // (5) Allocate Arena
  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // (6) Define Model Inputs
  // Obtain pointers to the model's input and output tensors.
  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
  if ((input->dims->size != 3)  
      || (input->dims->data[0] != 1) 
      || (input->dims->data[1] != 20) 
      || (input->dims->data[2] != 8)
      || (input->type != kTfLiteInt8)
      ) {
    MicroPrintf("Bad input tensor parameters in model");
    return;
  }
  // TODO: read more at:
  // https://www.tensorflow.org/lite/api_docs/c/union/tf-lite-ptr-union#union_tf_lite_ptr_union
  model_input_buffer = input->data.int8;
  //model_input_buffer = input->data.f;
  output = interpreter->output(0);

  // (7) Set-up Main Loop
  // Keep track of how many inferences we have performed.
  inference_count = 0;
}

///////////////////////////////////////////////////////////////////////////////
//
// loop
//
///////////////////////////////////////////////////////////////////////////////

void loop() 
{
  int8_t x_quantized = 0;
  int8_t y_quantized = 0;
  float y_pred = 0.0;


  // (1) Input Data
  // here, we receive 20*8 float numbers from Python program runing on PC;
  float received_float = 0.0;
  int received_count = 0;   // set this to 160 when using fixed data buffer
  data_received_flag = false;    // set this to true when using fixed data buffer
  while (Serial.available() && received_count < BUFFER_SIZE) {
    data_received_flag = true;
    received_float = Serial.parseFloat();     // parse it to float numbers
    // store the byte to data_buffer array
    data_buffer[received_count] = received_float;
    received_count++; 
  }


  // if we received a datapoint with exactly 20*8=160 values, then
  // we pass that to the model to make a prediction;
  if (data_received_flag == true) {

    // this will be used inside Python program 
    Serial.print(received_count);
    Serial.print("\r\n"); 

    if (received_count == BUFFER_SIZE) {
      // for debug purposes only:
      // Serial.println("Received " + String(received_count) + " numbers:");
      // for (int j=0; j<BUFFER_SIZE; j++) {
      //  Serial.print(data_buffer[j]);
      //  if (j < BUFFER_SIZE-1) { Serial.print(" "); }
      // }
      // Serial.print("\r\n");
      // delay(100);

      // to calculate exe time of inference, we record here
      // start time of inference
      long int t1_start = millis();

      // Copy feature buffer to input tensor
      // this is similar to the "micro_speech" example;
      for (int i = 0; i < BUFFER_SIZE; i++) {
        // print to check the Arduino received data_buffer value
        // Serial.print(data_buffer[i]);
        // Serial.print("\r\n"); 

        // take all 160 values from data_buffer and place into model_input_buffer; 
        // (2) Pre-Process
        // but, first Quantize the input from floating-point to integer
        //0.00392156 <-> input->params.scale
        x_quantized = data_buffer[i] / input->params.scale + input->params.zero_point;   // data_buffer_fixed[i] 
        model_input_buffer[i] = x_quantized;

        // ############################################
        // no quantization
        // model_input_buffer[i] = data_buffer[i];
      }

      // check for scale and zero_pint values for inputs
      // Serial.print(input->params.scale);
      // Serial.print("\r\n");
      // Serial.print(input->params.zero_point);
      // Serial.print("\r\n");


      // (3) Use Model to Make Inference
      // Run inference, and report any error
      TfLiteStatus invoke_status = interpreter->Invoke();
      if (invoke_status != kTfLiteOk) {
        MicroPrintf("Invoke failed");
        return;
      }


      // (4) Post-Process
      // this is similar to "hello_world" example from tinyML book;
      // just get the predicted output; it is only one value;
      // Obtain the quantized output from model's output tensor

      y_quantized = output->data.int8[0];
      // Dequantize the output from integer to floating-point

      // check for scale and zero_pint values for outputs
      // Serial.print(output->params.scale);
      // Serial.print("\r\n");
      // Serial.print(output->params.zero_point);
      // Serial.print("\r\n");

      // output->params.scale <-> 2.02084231 (I calculated as 2.00392157)
      y_pred = (y_quantized - output->params.zero_point) * output->params.scale;

      // ###################################
      // no dequantization
      // y_pred = output->data.f[0];

      // end time of inference
      long int t2_end = millis();


      // (5) Action 
      // in the original "hello_world" example, we used a HandleOutput() 
      // defined inside arduino_output_handler.cpp
      // here, we simply send the result back to Python program running on PC;
      //Serial.print("Inference time: ");
      //Serial.print(t2_end - t1_start);
      //Serial.print("\r\n"); // "\r\n"

      //Serial.print("Prediction #");
      //Serial.print(inference_count);
      //Serial.print(" is: ");
      //Serial.print("\r\n"); // "\r\n"
      
      // print quantized result
      // Serial.print(y_quantized);
      // Serial.print("\r\n");

      Serial.print(y_pred);
      Serial.print("\r\n");
      
      // Increment the inference_counter;
      inference_count += 1;
      if (inference_count >= 10000) inference_count = 0;
    } else { 
      // ERROR: case when received_count != BUFFER_SIZE
    } // received_count == BUFFER_SIZE
  } // data_received_flag == true
  delay(50);
  
}
