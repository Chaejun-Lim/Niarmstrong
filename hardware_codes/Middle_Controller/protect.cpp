#include <Arduino.h>
#include <stdint.h>

#include "protect.h"

// 6개 관절의 연속 에러 횟수를 개별/독립적으로 저장하는 전역 배열
static uint8_t temp_err_count[6] = {0, 0, 0, 0, 0, 0};
static uint8_t curr_err_count[6] = {0, 0, 0, 0, 0, 0};

// 💡 20ms 주기를 측정하기 위한 타이머 변수 추가
static uint32_t last_check_time = 0; 

void Init_Buzzer(void) 
{
    DDRB |= 0b00100000; 
    TCCR1A = 0; 
    TCCR1B = 0;
    TCCR1A |= (1 << COM1A1) | (1 << WGM11); 
    TCCR1B |= (1 << WGM13) | (1 << WGM12) | (1 << CS11); 
    ICR1 = 1999; 
    OCR1A = 0; 
}

// 안전 감시 및 연속 에러 카운트 기반 알람 함수
void Check_Safety_And_Alarm(uint8_t actuator_num, float *temp, float limit_temp, float *current, float *limit_current) 
{
    uint32_t current_time = millis(); // 현재 시간(밀리초) 확인
    
    // 💡 20ms(0.02초)가 지났을 때만 아래 로직을 실행 (메인 루프 속도와 무관하게 고정)
    if (current_time - last_check_time >= 20) 
    {
        last_check_time = current_time; // 마지막 검사 시간을 현재 시간으로 갱신
        bool is_alarm = false;
        
        // 6개의 관절을 독립적으로 검사
        for(uint8_t i = 0; i < actuator_num; i++)
        {
            // 1. 온도 연속 에러 카운트 계산 (최대 5까지만 증가)
            if (temp[i] > limit_temp) {
                if (temp_err_count[i] < 5) temp_err_count[i]++;
            } else {
                temp_err_count[i] = 0; 
            }
            
            // 2. 전류 연속 에러 카운트 계산 (최대 5까지만 증가)
            if (current[i] > limit_current[i] || current[i] < -limit_current[i]) {
                if (curr_err_count[i] < 5) curr_err_count[i]++;
            } else {
                curr_err_count[i] = 0; 
            }
            
            // 3. 알람 발생 조건 판별 (5회 도달 시 = 20ms * 5 = 100ms(0.1초))
            if (temp_err_count[i] >= 5 || curr_err_count[i] >= 5) {
                is_alarm = true;
            }
        }
        
        // 4. 상태에 따른 부저 동작 제어
        if (is_alarm)
        {
            OCR1A = 1000; // 50% 듀티비 인가 (삐이이이- 연속음 발생)
        } 
        else 
        {
            OCR1A = 0;    // 0% 듀티비 인가 (정상 상태일 때 즉시 소리 꺼짐)
        }
    }
}