
//=====================================================================================================
//  Include
//=====================================================================================================
#include "uart.h"
#include "can.h"
#include "gripper.h"
#include "calibration.h"
#include "protect.h"




//=====================================================================================================
//  Macro 정의
//=====================================================================================================
#define ACTUATOR_NUM       6U                              // 모든 함수에서 Gripper 개수는 1로 가정함

#define LPF_REF_FC         10.0F                           // [Hz]
#define LPF_REF_TSAMP      0.02F                           // [s]     

#define UART_TIMEOUT       3.0F                            // [s]
#define MICRO_UART_TIMEOUT (uint32_t)(UART_TIMEOUT * 1e6)  // [us]

#define DRIVE_TIME         LPF_REF_TSAMP                   // [s]
#define MICRO_DRIVE_TIME   (uint32_t)(LPF_REF_TSAMP * 1e6) // [us]

#define PRINT_TIME         1.0F                            // [s]
#define MICRO_PRINT_TIME   (uint32_t)(PRINT_TIME * 1e6)    // [us]

#define INIT_POS_LIMIT     20.0f                           // [degree]
#define USUAL_POS_LIMIT    140.0f                          // [degree]

#define VEL_LIMIT          20000                           // Raw Data in CAN
#define ACC_LIMIT          30000                           // Raw Data in CAN




//=====================================================================================================
//  전역 변수 선언
//=====================================================================================================
float coeff_ref;
uint8_t break_flag = 0U;

uint32_t NextUartTimeout;
uint32_t NextDriveTime;
uint32_t NextPrintTime;

float Received_PosRef[(ACTUATOR_NUM + 1U)] = {0.0f};
float PosRef_Filetered[(ACTUATOR_NUM + 1U)] = {0.0f};
float PosRef_for_Drive[(ACTUATOR_NUM + 1U)] = {0.0f};

const int16_t Vel_Limit_Raw[ACTUATOR_NUM] =
{
    VEL_LIMIT,  // J1
    VEL_LIMIT,  // J2
    VEL_LIMIT,  // J3
    VEL_LIMIT,  // J4
    VEL_LIMIT,  // J5
    2200    // J6
}; 

const int16_t Acc_Limit_Raw[ACTUATOR_NUM] =
{
    ACC_LIMIT, // J1
    ACC_LIMIT, // J2
    ACC_LIMIT, // J3
    ACC_LIMIT, // J4
    ACC_LIMIT, // J5
    30000  // J6
}; 

float Received_Pos[ACTUATOR_NUM]            = {0.0f}; 
float Current[ACTUATOR_NUM]                 = {0.0f};  
float Temp[ACTUATOR_NUM]                    = {0.0f};  
float Pos_for_Feedback[(ACTUATOR_NUM + 1U)] = {0.0f}; 

//  부저 알람(Safety Alarm) 제어용 함수 및 제한값 설정
const float Limit_Temp = 85.0f; // 모든 모터의 온도 제한값 (℃)

// 각 모터별 전류 제한값 (A) 설정 : { J1, J2, J3, J4, J5, J6 }
// 순서대로 J1~J3(2.0A), J4(2.1A), J5(2.7A)이며 J6은 2.0A.
const float Limit_Current[ACTUATOR_NUM] = {2.0f, 2.0f, 2.0f, 2.1f, 2.7f, 2.0f}; 




//=====================================================================================================
//  사용자 함수 Prototype
//=====================================================================================================
float LPF_CalcCoeff(float fc, float ts);
void LPF(float input, float *output, float coeff);




//=====================================================================================================
//  Setup 함수
//=====================================================================================================
void setup() 
{
    // 초기화 함수 호출
    Serial.begin(115200);
    pinMode(9, OUTPUT);
    Init_CAN();
    Init_Gripper();
    Init_Buzzer();
    if(!Init_UART1(76800)) 
    {
        Serial.println("Baud Rate를 9600/14400/19200/115200 중 하나로 설정하시오");
        Serial.println();
        while(1); 
    }
    sei();


    // LPF Coefficient 설정
    coeff_ref = LPF_CalcCoeff(LPF_REF_FC, LPF_REF_TSAMP);


    // Calibration Offset 측정 및 적용 확인 by Serial Monitor
    NextPrintTime = micros() + MICRO_PRINT_TIME;

    while(1)
    {
        if(Rx_Ok_Flag) 
        {
            if(!Parse_PosRef(Received_PosRef, ACTUATOR_NUM))
            {
                while(1)
                {
                    Serial.println("parsing error 발생");
                    Serial.println();
                }
            }

            Rx_Ok_Flag = false;
        }

        Read_Actuator_State(Received_Pos, Current, Temp, ACTUATOR_NUM);

        if ((NextPrintTime-micros()) & 0x80000000) 
        {
            uint32_t now = micros();

            Serial.println("Drive 전 출력");
            Print_Parsed_PosRef(Received_PosRef, ACTUATOR_NUM);
            Subtract_Pos_Offset(Received_Pos, Pos_for_Feedback, (ACTUATOR_NUM-1U)); // J6는 자체 Calibration
            Pos_for_Feedback[(ACTUATOR_NUM - 1)] = Received_Pos[(ACTUATOR_NUM - 1)];
            Print_Actuator_Pos(Pos_for_Feedback, ACTUATOR_NUM);

            NextPrintTime = now + MICRO_PRINT_TIME;
        }

        if (Serial.available() > 0)
        {
            char ch = Serial.read();

            if (ch == 'c')
            {
                Config_Pos_Offset(Received_Pos, (ACTUATOR_NUM-1));
            }
            else if (ch == 's')
            {
                Rx_Ok_Flag = false;
                break;
            }
        }
    }

    
    // 초기 위치 명령 안전 검사
    for(uint8_t i = 0; i < ACTUATOR_NUM; i++)
    {
        if(Received_PosRef[i] > INIT_POS_LIMIT || Received_PosRef[i] < -INIT_POS_LIMIT)
        {
            while(1)
            {
                Serial.println("리더암 값이 너무 크다 !!!!! 이멀전씨!!!");
                Serial.println();
            }
        }
    }


    // Timeout 설정 초기화
    uint32_t now = micros();
    NextUartTimeout = now + MICRO_UART_TIMEOUT;
    NextDriveTime = now + MICRO_DRIVE_TIME;
}




//=====================================================================================================
//  Main Loop 함수
//=====================================================================================================
void loop() 
{
    if(Rx_Ok_Flag) 
    {
//        digitalWrite(9, HIGH); // timing check using Oscilloscope

        if(Parse_PosRef(Received_PosRef, ACTUATOR_NUM))
        {
            // 위치 명령 정상 수신 시 next timeout 시점을 미루는 원리
            NextUartTimeout = micros() + MICRO_UART_TIMEOUT;
        }

        // J1 ~ J5는 센싱 값에 Calibration Offset 적용
        Subtract_Pos_Offset(Received_Pos, Pos_for_Feedback, (ACTUATOR_NUM-1U));

        // J6는 자체 Calibration -> 센싱 값 그대로 반환
        Pos_for_Feedback[(ACTUATOR_NUM-1U)] = Received_Pos[(ACTUATOR_NUM-1U)];

        // Gripper는 센서가 없으므로 Drive 값 그대로 반환
        Pos_for_Feedback[ACTUATOR_NUM] = PosRef_for_Drive[ACTUATOR_NUM];

        Send_Pos_Feedback(Pos_for_Feedback, ACTUATOR_NUM);

        Rx_Ok_Flag = false;

//        digitalWrite(9, LOW); // timing check using Oscilloscope
    }
    else 
    { 
        // CAN 통신을 통한 J1 ~ J6 상태 수신
        Read_Actuator_State(Received_Pos, Current, Temp, ACTUATOR_NUM);
    }


    if ((NextDriveTime-micros()) & 0x80000000) 
    {
        digitalWrite(9, HIGH); // timing check using Oscilloscope

        uint32_t now = micros();

        // J1 ~ Gripper Referene LPF 계산
        for(uint8_t i = 0; i < (ACTUATOR_NUM + 1U); i++)
        {
            LPF(Received_PosRef[i], &PosRef_Filetered[i], coeff_ref);
        }

        // J1 ~ J5는 LPF 적용, Calibration Offset 적용
        Add_Pos_Offset(PosRef_Filetered, PosRef_for_Drive, (ACTUATOR_NUM - 1));

        // J6는 자체 LPF, 자체 Calibration
        PosRef_for_Drive[(ACTUATOR_NUM - 1)] = Received_PosRef[(ACTUATOR_NUM - 1)];

        // Gripper는 LPF 적용, 자체 Calibration
        PosRef_for_Drive[ACTUATOR_NUM] = PosRef_Filetered[ACTUATOR_NUM];

        // Drive Actuator & Gripper
        Drive_All_Actuator(PosRef_for_Drive, ACTUATOR_NUM, Vel_Limit_Raw, Acc_Limit_Raw, USUAL_POS_LIMIT);
        Drive_Gripper(PosRef_for_Drive[ACTUATOR_NUM]);       

        NextDriveTime = now + MICRO_DRIVE_TIME;

        digitalWrite(9, LOW); // timing check using Oscilloscope
    }


    // 과전류, 과온도 알림용 부저 작동
    Check_Safety_And_Alarm(ACTUATOR_NUM, Temp, Limit_Temp, Current, Limit_Current);


    // 강제 timeout 호출
    if (Serial.available() > 0)
    {
        char ch = Serial.read();

        if (ch == 'b')
        {
            break_flag = 1U;
        }
    }


    // UART timeout callback
    if (((NextUartTimeout-micros()) & 0x80000000) || break_flag) 
    {
        NextDriveTime = micros() + MICRO_DRIVE_TIME;

        while(1)
        {
            Serial.println("UART timeout 발생, 기존 위치 유지 무한루프 실행");
            Serial.println();

            if ((NextDriveTime-micros()) & 0x80000000) 
            {
                uint32_t now = micros();

                // Drive Actuator & Gripper
                Drive_All_Actuator(PosRef_for_Drive, ACTUATOR_NUM, Vel_Limit_Raw, Acc_Limit_Raw, USUAL_POS_LIMIT);
                Drive_Gripper(PosRef_for_Drive[ACTUATOR_NUM]);

                NextDriveTime = now + MICRO_DRIVE_TIME;
            }
            
            if (Serial.available() > 0)
            {
                char ch = Serial.read();

                // restart
                if (ch == 'r')
                {
                    break;
                }
            }
        }

        NextUartTimeout = micros() + MICRO_UART_TIMEOUT;
        break_flag = 0U;
    }
}




//=====================================================================================================
//  LPF(Backward Euler) 함수 정의
//=====================================================================================================
float LPF_CalcCoeff(float fc, float ts)
{
    float wc = 2.0f * PI * fc;
    return (wc * ts) / (1.0f + (wc * ts));
}

void LPF(float input, float *output, float coeff)
{
    *output = (1.0f - coeff) * (*output) + coeff * input;
}
