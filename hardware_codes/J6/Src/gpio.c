/**
  ******************************************************************************
  * @file    gpio.c
  * @author  Kwon Dohyeon
  * @brief   GPIO initialization module.
  *          This file provides functions to initialize and configure
  *          GPIO peripherals.
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "stm32f767xx.h"
#include "register_macro.h"
#include "gpio.h"


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
  * @brief  Initialize GPIO peripherals according to the required configuration
  * @param  None
  * @retval None
  */
void Init_GPIO(void)
{
	/*
	 * @ GPIO 용도
	 * 6-step commutation에서 3상 인버터의 하단 FET를 구동하기 위한 신호 생성
	 *
	 * @ GPIO 요구사항
	 * 1) 출력 타입				  : push-pull
	 * 2) 출력 속도 				  : very High Speed
	 * 3) pull-up / pull-down 저항 : 사용x
	 * 4) 초기 출력				  : Low
	 *
	 * @ 사용하는 Pin
	 * PE8(UBOT), PE10(VBOT), PE12(WBOT)
	 */

	/* (1) GPIO clock 활성화 : AHB1 						 */
	/* 	   필요성			   : Reference Manual 5.2.12 참고 */
	RCC->AHB1ENR |= BIT4;

	/* (2) 출력 타입 설정: push-pull (0) -> Reset Value, 설정 필요x */

	/* (3) 출력 속도 설정: very High Speed (11) */
	GPIOE->OSPEEDR |=  (BIT17 | BIT16	// PE8 (UBOT)
					|	BIT21 | BIT20	// PE10(VBOT)
					|	BIT25 | BIT24); // PE12(WBOT)

	/* (4) pull-up / pull-down 저항 설정: 사용x (00) -> Reset Value, 설정 필요x */

	/* (5) **주의** GPIO 초기 출력 설정 : 안전장치, 용도에 따라 안전한 초기 출력이 달라짐 */
	GPIOE->BSRR = BIT24; 				// PE8 (UBOT) Low
	GPIOE->BSRR = BIT26; 				// PE10(VBOT) Low
	GPIOE->BSRR = BIT28; 				// PE12(WBOT) Low

	/* (6) Pin 모드 설정: General purpose output mode (01) */
	GPIOE->MODER &= ~(BIT17 | BIT16	    // PE8 (UBOT)
				  |	  BIT21 | BIT20	    // PE10(VBOT)
				  |	  BIT25 | BIT24);   // PE12(WBOT)

	GPIOE->MODER |=  (BIT16			    // PE8 (UBOT)
				  |	  BIT20			    // PE10(VBOT)
				  |	  BIT24);		    // PE12(WBOT)

	/*
	 * @ GPIO 용도
	 * Oscilloscope를 통한 Timing Verification
	 *
	 * @ GPIO 요구사항
	 * 1) 출력 타입				  : push-pull
	 * 2) 출력 속도 				  : very High Speed
	 * 3) pull-up / pull-down 저항 : 사용x
	 * 4) 초기 출력				  : Low
	 *
	 * @ 사용하는 Pin
	 * PE14(Timing Verification Using Oscilloscope)
	 */

	/* (1) GPIO clock 활성화 : AHB1 						 */
	/* 	   필요성			   : Reference Manual 5.2.12 참고 */
	RCC->AHB1ENR |= BIT4;

	/* (2) 출력 타입 설정: push-pull (0) -> Reset Value, 설정 필요x */

	/* (3) 출력 속도 설정: very High Speed (11) */
	GPIOE->OSPEEDR |=  (BIT29 | BIT28);

	/* (4) pull-up / pull-down 저항 설정: 사용x (00) -> Reset Value, 설정 필요x */

	/* (5) **주의** GPIO 초기 출력 설정 : 안전장치, 용도에 따라 안전한 초기 출력이 달라짐 */
	GPIOE->BSRR = BIT30; // PE14(Timing Verification) Low

	/* (6) Pin 모드 설정: General purpose output mode (01) */
	GPIOE->MODER &= ~(BIT29 | BIT28);
	GPIOE->MODER |=  BIT28;

	/*
	 * @ GPIO 용도
	 * LED를 통한 Code 동작 확인
	 *
	 * @ GPIO 요구사항
	 * 1) 출력 타입				  : push-pull
	 * 2) 출력 속도 				  : very High Speed
	 * 3) pull-up / pull-down 저항 : 사용x
	 * 4) 초기 출력				  : High
	 *
	 * @ 사용하는 Pin
	 * PC6(FLT_LED)
	 */

	/* (1) GPIO clock 활성화 : AHB1 						 */
	/* 	   필요성			   : Reference Manual 5.2.12 참고 */
	RCC->AHB1ENR |= BIT2;

	/* (2) 출력 타입 설정: push-pull (0) -> Reset Value, 설정 필요x */

	/* (3) 출력 속도 설정: very High Speed (11) */
	GPIOC->OSPEEDR |=  (BIT13 | BIT12);

	/* (4) pull-up / pull-down 저항 설정: 사용x (00) -> Reset Value, 설정 필요x */

	/* (5) **주의** GPIO 초기 출력 설정 : 안전장치, 용도에 따라 안전한 초기 출력이 달라짐 */
	GPIOC->BSRR = BIT6;  // PC6(FLT_LED) High

	/* (6) Pin 모드 설정: General purpose output mode (01) */
	GPIOC->MODER &= ~(BIT13 | BIT12);
	GPIOC->MODER |=  BIT12;
}

/**
  * @brief  Reset selected GPIO outputs
  * @param  U_ReSet  Set to 1 to reset the U-phase GPIO output
  * @param  V_ReSet  Set to 1 to reset the V-phase GPIO output
  * @param  W_ReSet  Set to 1 to reset the W-phase GPIO output
  * @retval None
  */
void Reset_UVW_GPIO(uint8_t U_ReSet, uint8_t V_ReSet, uint8_t W_ReSet)
{
	if(U_ReSet == 1U)
	{
		GPIOE->BSRR = BIT24; // PE8 (UBOT) Low
	}
	if(V_ReSet == 1U)
	{
		GPIOE->BSRR = BIT26; // PE10(VBOT) Low
	}
	if(W_ReSet == 1U)
	{
		GPIOE->BSRR = BIT28; // PE12(WBOT) Low
	}
}

/**
  * @brief  Set selected GPIO outputs
  * @param  U_Set  Set to 1 to set the U-phase GPIO output
  * @param  V_Set  Set to 1 to set the V-phase GPIO output
  * @param  W_Set  Set to 1 to set the W-phase GPIO output
  * @retval None
  */
void Set_UVW_GPIO(uint8_t U_Set, uint8_t V_Set, uint8_t W_Set)
{
	if(U_Set == 1U)
	{
		GPIOE->BSRR = BIT8;  // PE8 (UBOT) High
	}
	if(V_Set == 1U)
	{
		GPIOE->BSRR = BIT10; // PE10(VBOT) High
	}
	if(W_Set == 1U)
	{
		GPIOE->BSRR = BIT12; // PE12(WBOT) High
	}
}


/* Private functions --------------------------------------------------------*/
/**
  * @brief
  * @param
  * @retval
  */
