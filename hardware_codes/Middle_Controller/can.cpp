
#include "can.h"
#include <SPI.h>
#include <mcp_can.h>

//=====================================================================================================
//  Macro 정의
//=====================================================================================================
// MCP2515와의 SPI 통신을 위한 CS 핀 번호
#define SPI_CS_PIN 53U

//  AK actuator CAN Function ID
typedef enum
{
    CAN_PACKET_SET_DUTY = 0U,
    CAN_PACKET_SET_CURRENT,
    CAN_PACKET_SET_CURRENT_BRAKE,
    CAN_PACKET_SET_RPM,
    CAN_PACKET_SET_POS,
    CAN_PACKET_SET_ORIGIN_HERE,
    CAN_PACKET_SET_POS_SPD,
    CAN_PACKET_SET_MIT = 8U,
} CAN_PACKET_ID;

#define REAL_TIME_FEEDBACK_ID 0x29U




//=====================================================================================================
//  전역 변수 선언
//=====================================================================================================
// MCP2515 Library Class 객체 생성
MCP_CAN CAN0(SPI_CS_PIN);




//=====================================================================================================
//  CAN 초기화 함수
//=====================================================================================================
void Init_CAN(void)
{
    if (CAN0.begin(MCP_ANY, CAN_1000KBPS, MCP_8MHZ) != CAN_OK)
    {
        while (1);
    }

    CAN0.setMode(MCP_NORMAL);
}




//=====================================================================================================
//  Tx 함수
//=====================================================================================================
void Drive_All_Actuator(float * PosRef, uint8_t actuator_num, int16_t * Vel_Limit_Raw, int16_t * Acc_Limit_Raw, float Pos_Limit)
{
    // 안전을 위한 Reference 제한
    for(uint8_t i = 0; i < actuator_num; i++)
    {
        if(PosRef[i] > Pos_Limit)
        {
            PosRef[i] = Pos_Limit;
        }
        else if(PosRef[i] < -Pos_Limit)
        {
            PosRef[i] = -Pos_Limit;
        }
    }

    // Drive Actuator
    for(uint8_t i = 0; i < actuator_num; i++)
    {
        Pos_Vel_Loop_Mode(i+1, PosRef[i], Vel_Limit_Raw, Acc_Limit_Raw);
    }
}

// Cubemars AK-series manual의 5.1.7 Position-velocity Mode 참고
void Pos_Vel_Loop_Mode(uint8_t Actuator_ID, float PosRef, int16_t * Vel_Limit_Raw, int16_t * Acc_Limit_Raw) 
{
    uint8_t CAN_data[8];

    // CAN ID packing
    uint32_t can_ID = ((uint32_t)CAN_PACKET_SET_POS_SPD << 8) | (uint32_t)Actuator_ID;

    // CAN Data packing
    int32_t pos_data = (int32_t)(PosRef * 10000.0f);
    CAN_data[0] = (uint8_t)(((uint32_t)pos_data >> 24) & 0xFF);
    CAN_data[1] = (uint8_t)(((uint32_t)pos_data >> 16) & 0xFF);
    CAN_data[2] = (uint8_t)(((uint32_t)pos_data >> 8 ) & 0xFF);
    CAN_data[3] = (uint8_t)( (uint32_t)pos_data        & 0xFF);

    uint8_t index = Actuator_ID - 1;

    uint16_t vel = (uint16_t)Vel_Limit_Raw[index];

    CAN_data[4] = (uint8_t)((vel >> 8) & 0xFFU);
    CAN_data[5] = (uint8_t)( vel       & 0xFFU);

    uint16_t acc = (uint16_t)Acc_Limit_Raw[index];

    CAN_data[6] = (uint8_t)((acc >> 8) & 0xFFU);
    CAN_data[7] = (uint8_t)( acc       & 0xFFU);

    // CAN Frame Transmit
    CAN0.sendMsgBuf(can_ID, 1U, 8U, CAN_data); // MCP_CAN Class의 멤버 함수
}




//=====================================================================================================
//  Rx 함수
//=====================================================================================================
// Cubemars AK-series manual의 5.2.1 Servo mode CAN upload message protocol 참고
void Read_Actuator_State(float * Pos, float * Current, float * Temp, uint8_t actuator_num)
{
    while (CAN0.checkReceive() == CAN_MSGAVAIL)
    {
        uint32_t can_id = 0U;
        uint8_t dlc = 0U;
        uint8_t data[8];

        CAN0.readMsgBuf(&can_id, &dlc, data);

        uint8_t function_id = (uint8_t)((can_id >> 8) & 0xFFU);
        uint8_t actuator_id = (uint8_t)(can_id & 0xFFU);

        if ((function_id == REAL_TIME_FEEDBACK_ID) && (dlc == 8U))
        {
            if ((actuator_id >= 1U) && (actuator_id <= actuator_num))
            {
                uint8_t index = (uint8_t)(actuator_id - 1U);

                // 위치 파싱
                Pos[index] = (float)((int16_t)(((uint16_t)data[0] << 8) | (uint16_t)data[1])) * 0.1f;

                // 전류 파싱
                Current[index] = (float)((int16_t)(((uint16_t)data[4] << 8) | (uint16_t)data[5])) / 100.0f;

                // 온도 파싱
                Temp[index] = (float)((int8_t)data[6]);
            }
        }
    }
}

void Print_Actuator_Pos(float * Pos, uint8_t actuator_num)
{
    Serial.println("[Position Feedback]");

    for (uint8_t i = 0U; i < actuator_num; i++)
    {
        Serial.print("J");
        Serial.print(i + 1U);
        Serial.print(" Pos:");
        Print_Float_Width(Pos[i], 1U, 8U);
        Serial.println(" deg");
    }

    Serial.println();
}

void Print_Float_Width(float value, uint8_t decimal, uint8_t width)
{
    char buf[16];
    dtostrf(value, width, decimal, buf);
    Serial.print(buf);
}
