
#include "uart.h"
#include <Arduino.h>
#include "define.h"

//=====================================================================================================
//  Macro 정의
//=====================================================================================================
// UART Baud Rate
#define UBRR_9600               103U // 1ms per byte (8N1 기준)
#define UBRR_14400               68U 
#define UBRR_19200               51U 
#define UBRR_28800               34U 
#define UBRR_38400               25U 
#define UBRR_57600               16U 
#define UBRR_76800               12U 
#define UBRR_115200               8U 

// Frame Header Size: Start 4 byte, Command 1 byte, DLC 1 byte = 6 byte
#define FRAME_HEADER_SIZE         6U

// CRC Size: 2 byte
#define CRC_SIZE                  2U

// Start Marker 4byte
#define FRAME_START_0          0x7FU
#define FRAME_START_1          0xFFU
#define FRAME_START_2          0xFFU
#define FRAME_START_3          0xFFU

// Command 1 byte
typedef enum
{
    FRAME_CMD_POSREF = 1U, 
    FRAME_CMD_FEEDBACK,    
} FRAME_COMMAND_ID;

// Rx, Tx buffer size -> Command마다 DLC가 다르고,
// actuator 개수가 가변적이라고 가정하면 같은 Command에서도 DLC가 다를 수 있으므로 넉넉하게 설정
#define UART_BUFFER_SIZE_MAX  50U 




//=====================================================================================================
//  전역 변수 선언
//=====================================================================================================
// Rx, Tx Buffer
uint8_t rx_buf[UART_BUFFER_SIZE_MAX];    
uint8_t rx_cpy_buf[UART_BUFFER_SIZE_MAX];   
uint8_t tx_buf[UART_BUFFER_SIZE_MAX];

// CRC Value
uint8_t RX_CRCHi;
uint8_t RX_CRCLo;
uint8_t TX_CRCHi;
uint8_t TX_CRCLo;

// Rx complete flag
bool Rx_Ok_Flag = false;




//=====================================================================================================
//  UART 초기화 함수
//
//  정해진 baud rate들 중 하나를 parameter로 받아서 설정 가능하다. 
//  frame은 8N1으로 고정이며, Rx interrupt를 활성화한다.
//=====================================================================================================
bool Init_UART1(uint32_t baud_rate)
{
    // configure baudrate
    if(baud_rate == 9600)
    {
        UBRR1H = (uint8_t)(UBRR_9600 >> 8);
        UBRR1L = (uint8_t)(UBRR_9600 & 0xFFU);
    }
    else if(baud_rate == 14400)
    {
        UBRR1H = (uint8_t)(UBRR_14400 >> 8);
        UBRR1L = (uint8_t)(UBRR_14400 & 0xFFU);
    }
    else if(baud_rate == 19200)
    {
        UBRR1H = (uint8_t)(UBRR_19200 >> 8);
        UBRR1L = (uint8_t)(UBRR_19200 & 0xFFU);
    }
    else if(baud_rate == 28800)
    {
        UBRR1H = (uint8_t)(UBRR_28800 >> 8);
        UBRR1L = (uint8_t)(UBRR_28800 & 0xFFU);
    }
    else if(baud_rate == 38400)
    {
        UBRR1H = (uint8_t)(UBRR_38400 >> 8);
        UBRR1L = (uint8_t)(UBRR_38400 & 0xFFU);
    }
    else if(baud_rate == 57600)
    {
        UBRR1H = (uint8_t)(UBRR_57600 >> 8);
        UBRR1L = (uint8_t)(UBRR_57600 & 0xFFU);
    }
    else if(baud_rate == 76800)
    {
        UBRR1H = (uint8_t)(UBRR_76800 >> 8);
        UBRR1L = (uint8_t)(UBRR_76800 & 0xFFU);
    }
    else if(baud_rate == 115200)
    {
        UBRR1H = (uint8_t)(UBRR_115200 >> 8);
        UBRR1L = (uint8_t)(UBRR_115200 & 0xFFU);
    }
    else
    {
        return false;
    }
    
    // RX enable, TX enable, RX complete interrupt enable
    UCSR1B |= (1U << RXEN1) | (1U << TXEN1);
    UCSR1B |= (1U << RXCIE1);

    // configure frame : 8N1
    UCSR1C |= (1U << UCSZ11) | (1U << UCSZ10);   

    return true; 
}




//=====================================================================================================
//  Rx 함수
//
//  interrupt로 하나의 Frame을 검증(CRC: Modbus CRC16) 및 저장하고, 
//  Parsing을 통해 7개 모터의 위치 지령을 배열에 저장한다.
//=====================================================================================================
static uint8_t auchCRCHi[]= {
0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81,
0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0,
0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01,
0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81,
0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0,
0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01,
0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81,
0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0,
0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01,
0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81,
0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0,
0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01,
0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81,
0x40
};

static uint8_t auchCRCLo[]= {
0x00, 0xC0, 0xC1, 0x01, 0xC3, 0x03, 0x02, 0xC2, 0xC6, 0x06, 0x07, 0xC7, 0x05, 0xC5, 0xC4,
0x04, 0xCC, 0x0C, 0x0D, 0xCD, 0x0F, 0xCF, 0xCE, 0x0E, 0x0A, 0xCA, 0xCB, 0x0B, 0xC9, 0x09,
0x08, 0xC8, 0xD8, 0x18, 0x19, 0xD9, 0x1B, 0xDB, 0xDA, 0x1A, 0x1E, 0xDE, 0xDF, 0x1F, 0xDD,
0x1D, 0x1C, 0xDC, 0x14, 0xD4, 0xD5, 0x15, 0xD7, 0x17, 0x16, 0xD6, 0xD2, 0x12, 0x13, 0xD3,
0x11, 0xD1, 0xD0, 0x10, 0xF0, 0x30, 0x31, 0xF1, 0x33, 0xF3, 0xF2, 0x32, 0x36, 0xF6, 0xF7,
0x37, 0xF5, 0x35, 0x34, 0xF4, 0x3C, 0xFC, 0xFD, 0x3D, 0xFF, 0x3F, 0x3E, 0xFE, 0xFA, 0x3A,
0x3B, 0xFB, 0x39, 0xF9, 0xF8, 0x38, 0x28, 0xE8, 0xE9, 0x29, 0xEB, 0x2B, 0x2A, 0xEA, 0xEE,
0x2E, 0x2F, 0xEF, 0x2D, 0xED, 0xEC, 0x2C, 0xE4, 0x24, 0x25, 0xE5, 0x27, 0xE7, 0xE6, 0x26,
0x22, 0xE2, 0xE3, 0x23, 0xE1, 0x21, 0x20, 0xE0, 0xA0, 0x60, 0x61, 0xA1, 0x63, 0xA3, 0xA2,
0x62, 0x66, 0xA6, 0xA7, 0x67, 0xA5, 0x65, 0x64, 0xA4, 0x6C, 0xAC, 0xAD, 0x6D, 0xAF, 0x6F,
0x6E, 0xAE, 0xAA, 0x6A, 0x6B, 0xAB, 0x69, 0xA9, 0xA8, 0x68, 0x78, 0xB8, 0xB9, 0x79, 0xBB,
0x7B, 0x7A, 0xBA, 0xBE, 0x7E, 0x7F, 0xBF, 0x7D, 0xBD, 0xBC, 0x7C, 0xB4, 0x74, 0x75, 0xB5,
0x77, 0xB7, 0xB6, 0x76, 0x72, 0xB2, 0xB3, 0x73, 0xB1, 0x71, 0x70, 0xB0, 0x50, 0x90, 0x91,
0x51, 0x93, 0x53, 0x52, 0x92, 0x96, 0x56, 0x57, 0x97, 0x55, 0x95, 0x94, 0x54, 0x9C, 0x5C,
0x5D, 0x9D, 0x5F, 0x9F, 0x9E, 0x5E, 0x5A, 0x9A, 0x9B, 0x5B, 0x99, 0x59, 0x58, 0x98, 0x88,
0x48, 0x49, 0x89, 0x4B, 0x8B, 0x8A, 0x4A, 0x4E, 0x8E, 0x8F, 0x4F, 0x8D, 0x4D, 0x4C, 0x8C,
0x44, 0x84, 0x85, 0x45, 0x87, 0x47, 0x46, 0x86, 0x82, 0x42, 0x43, 0x83, 0x41, 0x81, 0x80,
0x40
};

void Crc_RTU_RX(uint8_t nsVal)
{
    uint8_t uIndex;

    uIndex = RX_CRCHi ^ nsVal;
    RX_CRCHi = RX_CRCLo ^ auchCRCHi[uIndex];
    RX_CRCLo = auchCRCLo[uIndex];
}

void Crc_RTU_TX(uint8_t nsVal)
{
    uint8_t uIndex;

    uIndex = TX_CRCHi ^ nsVal;
    TX_CRCHi = TX_CRCLo ^ auchCRCHi[uIndex];
    TX_CRCLo = auchCRCLo[uIndex];
}
   
void Make_CRC_RX(uint16_t limit)
{   
    static uint16_t i=0;

    RX_CRCHi = 0xFF;
    RX_CRCLo = 0xFF;
    for(i = 0; i < limit; i++)
    {
        Crc_RTU_RX(rx_buf[i]);
    }
}

void Make_CRC_TX(uint16_t limit)
{   
    static uint16_t i=0;

    TX_CRCHi = 0xFF;
    TX_CRCLo = 0xFF;
    for(i = 0; i < limit; i++)
    {
        Crc_RTU_TX(tx_buf[i]);
    }
}

ISR(USART1_RX_vect)
{
    static uint8_t buf_index = 0;

    // frame error(stop bit error)가 발생하지 않은 경우
    if(!(UCSR1A & BIT4))
    {
        // data overrun(buffer 덮어쓰기)가 발생하지 않은 경우
        if(!(UCSR1A & BIT3))
        {
            // read 1byte and write to rx buffer
            rx_buf[buf_index++] = UDR1;

            // Start Maker Filtering
            if(rx_buf[0] != FRAME_START_0) 
            {
                Serial.println("FRAME_START_0 불일치");
                Serial.println();
                buf_index = 0; return;
            }
            if(buf_index == 1) return;

            if(rx_buf[1] != FRAME_START_1) 
            {
                Serial.println("FRAME_START_1 불일치");
                Serial.println();
                buf_index = 0; return;
            }
            if(buf_index == 2) return;

            if(rx_buf[2] != FRAME_START_2) 
            {
                Serial.println("FRAME_START_2 불일치");
                Serial.println();
                buf_index = 0; return;
            }
            if(buf_index == 3) return;

            if(rx_buf[3] != FRAME_START_3) 
            {
                Serial.println("FRAME_START_3 불일치");
                Serial.println();
                buf_index = 0; return;
            }
            if(buf_index == 4) return;

            // Command, DLC는 저장만 하기
            if((buf_index == 5) || (buf_index == 6)) 
            {
                return;
            }

            // rx buffer에 CRC까지 저장된 경우
            if(buf_index == (FRAME_HEADER_SIZE + rx_buf[5] + CRC_SIZE)) 
            {
                // 수신 data에서 CRC 계산 대상: CRC를 제외한 전부
                Make_CRC_RX((FRAME_HEADER_SIZE + rx_buf[5]));
                
                // 계산한 CRC와 수신받은 CRC가 일치하는 경우
                if((rx_buf[(FRAME_HEADER_SIZE + rx_buf[5])] == RX_CRCHi) && (rx_buf[(FRAME_HEADER_SIZE + rx_buf[5] + 1)] == RX_CRCLo))
                {
                    // 이전 frame이 Parsing & Drive 후인 경우
                    if (!Rx_Ok_Flag)
                    {
                        Rx_Ok_Flag = true;

                        // CRC를 제외한 START + COMMAND + DLC + PAYLOAD 32 byte 복사
                        memcpy(&rx_cpy_buf[0], &rx_buf[0], (FRAME_HEADER_SIZE + rx_buf[5]));
                    }
                    // 이전 frame이 Parsing & Drive 전인 경우, 새 frame 무시
                    else
                    {
                        Serial.println("CRC 통과, 그러나 이전 frame의 처리가 완료되지 않아서 새로운 frame을 무시");
                        Serial.println();
                    }
                }
                // 계산한 CRC와 수신받은 CRC가 일치하지 않는 경우
                else
                {
                    Serial.println("CRC 불일치");
                    Serial.println();
                }

                buf_index = 0;
            }

            if(buf_index > (FRAME_HEADER_SIZE + rx_buf[5] + CRC_SIZE))
            {
                Serial.println("buffer index overflow 발생");
                Serial.println();
                buf_index = 0; return;
            }
        }
        // data overrun(buffer 덮어쓰기)이 발생한 경우
        else
        {
            Serial.println("data overrun 발생");
            Serial.println();
            uint8_t dummy = UDR1; // data overrun flag clear
            buf_index = 0; return;
        }
    }
    // frame error가 발생한 경우
    else
    {
        Serial.println("frame error 발생");
        Serial.println();
        uint8_t dummy = UDR1; // frame error flag clear
        buf_index = 0; return;
    }
}

// Cubemars AK-series manual의 5.1.7 Position-velocity Mode 참고
bool Parse_PosRef(float * PosRef, uint8_t actuator_num)
{
    // 수신받은 Frame의 Command가 Position Reference Command가 아니면 Parsing하지 않음
    if(rx_cpy_buf[4] != FRAME_CMD_POSREF)
    {
        return false;
    }

    // 함수에 인가한 Actuator 개수와 수신 받은 Actuator의 위치 Reference 개수가 다르면 Parsing하지 않음
    // DLC에서 Gripper Reference 2 byte를 제외한 뒤, 4 byte로 나눔
    if(((rx_cpy_buf[5] - 2U) / 4U) != actuator_num)
    {
        return false;
    }

    // Parsing 후 parameter로 입력 받은 배열 주소에 저장
    for (uint8_t i = 0U; i < actuator_num; i++)
    {
        uint8_t actuator_offset = (uint8_t)(6U + (i * 4U));

        uint32_t actuator_raw = 0U;
        actuator_raw |= ((uint32_t)rx_cpy_buf[actuator_offset]      << 24);
        actuator_raw |= ((uint32_t)rx_cpy_buf[actuator_offset + 1U] << 16);
        actuator_raw |= ((uint32_t)rx_cpy_buf[actuator_offset + 2U] <<  8);
        actuator_raw |= ((uint32_t)rx_cpy_buf[actuator_offset + 3U]      );

        PosRef[i] = (float)((int32_t)actuator_raw) / 10000.0f;
    }

    uint8_t gripper_offset = (uint8_t)(6U + actuator_num * 4U);

    uint16_t gripper_raw = 0U;
    gripper_raw |= ((uint16_t)rx_cpy_buf[gripper_offset] << 8);
    gripper_raw |= ((uint16_t)rx_cpy_buf[gripper_offset + 1U]);

    PosRef[actuator_num] = (float)gripper_raw / 65535.0f;

    return true;
}

void Print_Parsed_PosRef(float * PosRef, uint8_t actuator_num)
{
    Serial.println(F("[Parsed Position Reference]"));

    for (uint8_t i = 0U; i < actuator_num; i++)
    {
        Serial.print(F("J"));
        Serial.print(i + 1U);
        Serial.print(F(" = "));
        Serial.print(PosRef[i], 4);
        Serial.println(F(" deg"));
    }

    Serial.print(F("Gripper = "));
    Serial.println(PosRef[actuator_num], 4);

    Serial.println();
}




//=====================================================================================================
//  Tx 함수
//  7개 모터의 상태 배열의 시작 주소를 parameter로 받아서 UART를 통해 순차적으로 송신한다.
//=====================================================================================================
// Cubemars AK-series manual의 5.2.1 Servo mode CAN upload message protocol 참고
void Send_Pos_Feedback(float * Pos, uint8_t actuator_num)
{
    tx_buf[0] = FRAME_START_0;
    tx_buf[1] = FRAME_START_1;
    tx_buf[2] = FRAME_START_2;
    tx_buf[3] = FRAME_START_3;
    tx_buf[4] = FRAME_CMD_FEEDBACK;
    tx_buf[5] = (actuator_num + 1U) * 2U;

    for (uint8_t i = 0U; i < actuator_num; i++)
    {
        uint16_t actuator_raw = (uint16_t)((int16_t)(Pos[i] * 10.0f));

        tx_buf[(6U + i*2U)] = (uint8_t)((actuator_raw >> 8) & 0xFFU);
        tx_buf[(7U + i*2U)] = (uint8_t)( actuator_raw       & 0xFFU);
    }

    uint16_t gripper_raw = (uint16_t)(Pos[actuator_num] * 65535.0f);

    tx_buf[(actuator_num * 2U + 6U)] = (uint8_t)((gripper_raw >> 8) & 0xFFU);
    tx_buf[(actuator_num * 2U + 7U)] = (uint8_t)( gripper_raw       & 0xFFU);

    // CRC 계산 대상: START + COMMAND + DLC + PAYLOAD
    Make_CRC_TX((FRAME_HEADER_SIZE + (actuator_num + 1U) * 2U));

    tx_buf[(FRAME_HEADER_SIZE + (actuator_num + 1U) * 2U)] = TX_CRCHi;
    tx_buf[(FRAME_HEADER_SIZE + (actuator_num + 1U) * 2U + 1)] = TX_CRCLo;

    for (uint8_t i = 0U; i < (FRAME_HEADER_SIZE + (actuator_num + 1U) * 2U + CRC_SIZE); i++)
    {
        Send_Byte(tx_buf[i]);
    }
}

void Send_Byte(uint8_t data)
{
    // TX data register empty 대기.
    while ((UCSR1A & (1U << UDRE1)) == 0U);
    UDR1 = data;
}
