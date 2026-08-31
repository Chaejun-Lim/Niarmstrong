
#include "calibration.h"

float Pos_Offset[10] = {0.0f}; // actuator 개수가 가변적이라고 가정하고 넉넉하게 설정

void Config_Pos_Offset(float * Pos, uint8_t actuator_num)
{
    for(uint8_t i = 0; i < actuator_num; i++)
    {
        Pos_Offset[i] = Pos[i];
    }
}

void Add_Pos_Offset(float * Pos_Raw, float * Pos, uint8_t actuator_num)
{
    // Calibration Offset 적용
    for(uint8_t i = 0; i < actuator_num; i++)
    {
        Pos[i] = Pos_Raw[i] + Pos_Offset[i];
    }
}

void Subtract_Pos_Offset(float * Pos_Raw, float * Pos, uint8_t actuator_num)
{
    // Calibration Offset 적용
    for(uint8_t i = 0; i < actuator_num; i++)
    {
        Pos[i] = Pos_Raw[i] - Pos_Offset[i];
    }
}