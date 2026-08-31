
#ifndef INC_CAN_H_
#define INC_CAN_H_

#include <Arduino.h>

//=====================================================================================================
//  함수 prototype
//=====================================================================================================
void Init_CAN(void);

void Drive_All_Actuator(float * PosRef, uint8_t actuator_num, int16_t * Vel_Limit_Raw, int16_t * Acc_Limit_Raw, float Pos_Limit);
void Pos_Vel_Loop_Mode(uint8_t Actuator_ID, float PosRef, int16_t * Vel_Limit_Raw, int16_t * Acc_Limit_Raw);

void Read_Actuator_State(float * Pos, float * Current, float * Temp, uint8_t actuator_num);
void Print_Actuator_Pos(float * Pos, uint8_t actuator_num);
void Print_Float_Width(float value, uint8_t decimal, uint8_t width);

#endif /* INC_CAN_H_ */
