/**
  ******************************************************************************
  * @file    adc.c
  * @author  Kwon Dohyeon
  * @brief   ADC initialization module.
  *          This file provides functions to initialize and configure
  *          ADC peripherals.
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "stm32f767xx.h"
#include "register_macro.h"
#include "adc.h"


/* Private typedef -----------------------------------------------------------*/
/*  */


/* Private define ------------------------------------------------------------*/
/*  */


/* Private macros ------------------------------------------------------------*/
/*  */


/* Private variables ---------------------------------------------------------*/
/*  */


/* Private function prototypes -----------------------------------------------*/
/*  */


/* Exported functions --------------------------------------------------------*/
/**
  * @brief  Initialize ADC peripherals according to the required configuration
  * @param  None
  * @retval None
  */
void Init_ADC(void)
{
	/*
	 * @ ADC 용도
	 * 3상 전류 센싱
	 *
	 * @ ADC 요구사항
	 * 1)3상 전류를 같은 타이밍에 센싱하기 위해 Tripple Injected Simultaneous Mode 사용 필수
	 * 2)전류 ripple의 영향을 피해서 평균 전류를 측정하기 위해 TIM1 TRGO 트리거를 Conversion Source로 사용
	 * 3)
	 *
	 * @ 사용 Pin
	 * PA0(ADC1_IN0, A상), PA1(ADC2_IN1, B상), PA2(ADC3_IN2, C상)
	 */

    /* (1) GPIO 설정 */
    /* PA0~PA3, PA6~PA7: 아날로그 모드 (MODER = 11)      */
    /* PA4, PA5는 DAC 출력으로 사용 (dac.c에서 설정)        */
	/* GPIO clock 활성화 : AHB1 						 */
	/* 필요성			   : Reference Manual 5.2.12 참고 */
	RCC->AHB1ENR |= BIT0;
    GPIOA->MODER |= GPIO_MODER_MODER0 | GPIO_MODER_MODER1 | GPIO_MODER_MODER2
                  | GPIO_MODER_MODER3 | GPIO_MODER_MODER6 | GPIO_MODER_MODER7;

    /* (2) ADC 클럭 활성화 */
    RCC->APB2ENR |= RCC_APB2ENR_ADC1EN | RCC_APB2ENR_ADC2EN | RCC_APB2ENR_ADC3EN;

    /* (3) ADC 공통 설정 */
    /* MULTI[4:0] = 10101: Triple Injected Simultaneous Mode */
    /* ADCPRE = 00: PCLK2/2 = 54MHz/2 = 27MHz				 */
    ADC->CCR = ADC_CCR_MULTI_4 | ADC_CCR_MULTI_2 | ADC_CCR_MULTI_0;

    /* (4) ADC1 설정 (PA0 = CH0 = ias) */
    /* SMPR2: CH0~CH7 샘플링 시간 = 56 cycles (각 채널 3비트 = 0b011) */
    ADC1->SMPR2 = 0x006DB6DB;
    ADC1->CR1 = ADC_CR1_SCAN;         // Scan 모드 활성화
    ADC1->CR2 = ADC_CR2_ADON;         // ADC 활성화
    ADC1->SQR1 = 0;                   // 레귤러 시퀀스: 1개 변환 (L=0)
    ADC1->JSQR = 0;                   // 인젝티드 시퀀스: CH0 (JL=0, JSQ4=0)

    /* (5) ADC2 설정 (PA1 = CH1 = ibs) */
    ADC2->SMPR2 = 0x006DB6DB;         // CH0~CH7 = 56 cycles
    ADC2->CR1 = ADC_CR1_SCAN;
    ADC2->CR2 = ADC_CR2_ADON;
    ADC2->SQR1 = 0;                   // 레귤러 시퀀스: 1개 변환
    ADC2->JSQR = (1U << ADC_JSQR_JSQ4_Pos);  // 인젝티드 시퀀스: CH1 (JSQ4=1)

    /* (6) ADC3 설정 (PA2 = CH2 = ics) */
    ADC3->SMPR2 = 0x006DB6DB;         // CH0~CH7 = 56 cycles
    ADC3->CR1 = ADC_CR1_SCAN;
    ADC3->CR2 = ADC_CR2_ADON;
    ADC3->SQR1 = 0;                   // 레귤러 시퀀스: 1개 변환
    ADC3->JSQR = (2U << ADC_JSQR_JSQ4_Pos);  // 인젝티드 시퀀스: CH2 (JSQ4=2)
}


/* Private functions --------------------------------------------------------*/
/**
  * @brief
  * @param
  * @retval
  */

