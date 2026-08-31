
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <EEPROM.h>
#include "gripper.h"

// Class 객체 생성
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVO_CHANNEL 15  

#define INPUT_MAX 0.27F

#define GRIPPER_OPEN 2500.0F
#define GRIPPER1_CLOSED 450.0F

int targetPulseWidth;        

void Init_Gripper() 
{
    pwm.begin();
    pwm.setPWMFreq(50); // Servo Motor에 인가하는 PWM 주파수 [Hz] = 20000 [us], 절대 바꾸지 말 것
}

// 입력: 0 ~ INPUT_MAX
void Drive_Gripper(float gripper_Ref)
{    
    // 비율을 Pulse Width로 변환 (0.0 -> GRIPPER_OPEN, INPUT_MAX -> GRIPPER1_CLOSED로 선형 변환) : 실험으로 구한 범위
    float targetPulseWidth = ((GRIPPER1_CLOSED - GRIPPER_OPEN) / INPUT_MAX) * gripper_Ref + GRIPPER_OPEN;

    // 펄스 폭 staturation
    if (targetPulseWidth < GRIPPER1_CLOSED)  targetPulseWidth = GRIPPER1_CLOSED;
    if (targetPulseWidth > GRIPPER_OPEN)     targetPulseWidth = GRIPPER_OPEN;

    // PWM pulse width 인가
    pwm.writeMicroseconds(SERVO_CHANNEL, targetPulseWidth);
}
