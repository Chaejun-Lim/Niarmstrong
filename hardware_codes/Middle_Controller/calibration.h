
#ifndef INC_CALIBRATION_H_
#define INC_CALIBRATION_H_

#include <Arduino.h>

//=====================================================================================================
//  함수 prototype
//=====================================================================================================
void Config_Pos_Offset(float * Pos, uint8_t actuator_num);
void Add_Pos_Offset(float * Pos_Raw, float * Pos, uint8_t actuator_num);
void Subtract_Pos_Offset(float * Pos_Raw, float * Pos, uint8_t actuator_num);

#endif /* INC_CALIBRATION_H_ */