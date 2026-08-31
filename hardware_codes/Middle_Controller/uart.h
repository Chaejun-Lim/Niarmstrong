
#ifndef INC_UART_H_
#define INC_UART_H_

#include <Arduino.h>

//=====================================================================================================
//  외부 전역 변수 선언
//=====================================================================================================
extern bool Rx_Ok_Flag;

//=====================================================================================================
//  함수 prototype
//=====================================================================================================
bool Init_UART1(uint32_t baud_rate);

bool Parse_PosRef(float * PosRef, uint8_t actuator_num);
void Print_Parsed_PosRef(float * PosRef, uint8_t actuator_num);

void Send_Pos_Feedback(float * Pos, uint8_t actuator_num);
void Send_Byte(uint8_t data);

#endif /* INC_UART_H_ */
