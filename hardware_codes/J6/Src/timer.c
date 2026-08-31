/**
  ******************************************************************************
  * @file    timer.c
  * @author  Kwon Dohyeon
  * @brief   Timer initialization module.
  *          This file provides functions to initialize and configure
  *          Timer peripherals.
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "stm32f767xx.h"
#include "register_macro.h"
#include "timer.h"


/* Private typedef -----------------------------------------------------------*/
/*  */


/* Private define ------------------------------------------------------------*/
/* PWM define */
#define FPWM			20000UL			 // [Hz]
#define FCLK			216000000UL		 // [Hz]
#define CNT_TOP 		(FCLK/(2U*FPWM)) // PWM을 만드는 카운터의 TOP


/* Private macros ------------------------------------------------------------*/
/*  */


/* Private variables ---------------------------------------------------------*/
/*  */


/* Private function prototypes -----------------------------------------------*/
/*  */


/* Exported functions --------------------------------------------------------*/
/**
  * @brief  Initialize PWM generation according to the required configuration
  * @param  None
  * @retval None
  */
void Init_UVW_PWM(void)
{
	/*
	 * @ Timer 용도
	 * 3상 인버터의 상단 FET 구동을 위한 PWM 신호 생성
	 *
	 * @ Timer 요구사항
	 * 1) 3개의 PWM 신호 생성
	 * 2) PWM 주파수는 20kHz
	 * 3) duty ratio와 관계 없이 Off duty의 중앙에서 전류 센싱을 위한 ADC를 동작시키기 위해
	 *    center-aligned mode 필요하며, off duty 중앙에서 ADC trigger 신호를 발생시켜야 함
	 * 4) disable 시 High-Z가 아닌 Low 신호가 출력되도록 설정, enable과 disable을
	 *    자유자재로할 수 있어야 함
	 *
	 * @ 사용하는 Pin
	 * PE9(UTOP, TIM1_CH1), PE11(VTOP, TIM1_CH2), PE13(WTOP, TIM1_CH3)
	 */

	/* (1) GPIO 클럭 활성화 : AHB1(216MHz)				   */
	/* 	   필요성			 : Reference Manual 5.2.12 참고 */
	RCC->AHB1ENR |= BIT4;

	/* (2) Alternate Function 설정: AF1 */
	GPIOE->AFR[1]  &= ~(BIT23 | BIT22 | BIT21 | BIT20 	// PE13(WTOP, TIM1_CH3)
    				|   BIT15 | BIT14 | BIT13 | BIT12   // PE11(VTOP, TIM1_CH2)
					|   BIT7  | BIT6  | BIT5  | BIT4 ); // PE9(UTOP, TIM1_CH1)

	GPIOE->AFR[1]  |=  (BIT20							// PE13(WTOP, TIM1_CH3)
					|   BIT12							// PE11(VTOP, TIM1_CH2)
					|   BIT4 ); 						// PE9(UTOP, TIM1_CH1)

	/* (3) 출력 속도 설정: very High Speed (11) */
	GPIOE->OSPEEDR |=  (BIT27 | BIT26 					// PE13(WTOP, TIM1_CH3)
					|   BIT23 | BIT22   				// PE11(VTOP, TIM1_CH2)
					|   BIT19 | BIT18); 				// PE9(UTOP, TIM1_CH1)

	/* (4) 출력 타입 설정: push-pull (00) -> Reset Value, 설정 필요x */

	/* (5) pull-up / pull-down 저항 설정: 사용x (00) -> Reset Value, 설정 필요x */

	/* (6) Pin 모드 설정: General purpose output mode (01) */
	GPIOE->MODER   &= ~(BIT27 | BIT26 					// PE13(WTOP, TIM1_CH3)
					|   BIT23 | BIT22   				// PE11(VTOP, TIM1_CH2)
					|   BIT19 | BIT18); 				// PE9(UTOP, TIM1_CH1)

	GPIOE->MODER   |=  (BIT27							// PE13(WTOP, TIM1_CH3)
					|   BIT23							// PE11(VTOP, TIM1_CH2)
					|   BIT19); 						// PE9(UTOP, TIM1_CH1)

	/* (7) TIM1 클럭 활성화 */
	RCC->APB2ENR |= BIT0; // TIM1

	/* (8) 카운터 기본 설정: 카운터는 channel에 관계 없이 공통 -> 카운터 클럭, TOP 모두 공통 */
	TIM1->PSC = 0u;		 								// Prescaler, 분주비 = (PSC + 1) = 1 (216MHz 그대로 사용)
	TIM1->ARR = CNT_TOP; 								// counter의 top 값
	TIM1->CCR1 = 0u;
	TIM1->CCR2 = 0u;
	TIM1->CCR3 = 0u;     								// compare register 초기화

	/*
	 * (9) PWM 모드 설정 -> OCx, OCxN이 아닌 이들의 원신호인 OCxREF의 행동을 정의
	 * 					  Reference Manual Figure 191 참고
	 *
	 * @ PWM Mode 2
	 * in upcounting   : CNT >= CCR → active
	 * in downcounting : CNT >  CCR → active
	 * OCxREF는 active High
	 * Reference Manual 25.4.8 참고
	 *
	 * >=과 >이 카운팅 방향에 따라 다르다는 것은 Reference Manual 25.4.8 뿐만 아니라
	 * Reference Manual Figure 226 에서도 확인 가능
	 *
	 * 실제 출력 OCx, OCxN은 Reference Manual Table 179 참고
	 * 또한, CCR은 wirte 후 즉각적으로 반영되지 않고 UEV가 발생했을 떄만 반영되도록 설정하여(preload)
	 * PWM의 한 주기 중간에 duty 비가 바뀌는 것을 방지함.
	 */
	TIM1->CCMR1 |=  (BIT6  | BIT5  | BIT4  				// CH1 PWM mode 2(OC1M)
				 |   BIT3				   				// CH1 preload enable(OC1PE)
				 |   BIT14 | BIT13 | BIT12 				// CH2 PWM mode 2(OC2M)
				 |   BIT11);			   				// CH2 preload enable(OC2PE)

	TIM1->CCMR2 |=  (BIT6  | BIT5  | BIT4  				// CH3 PWM mode 2(OC3M)
				|    BIT3 );			   				// CH3 preload enable(OC3PE)

	/* (10) 초기에는 Reference Manual Table 179의 1번으로 설정, 따라서 *** Low 신호 출력 *** */
	TIM1->CCER &= ~BIT1;  								// OC1 active High
	TIM1->CCER &= ~BIT5;  								// OC2 active High
	TIM1->CCER &= ~BIT9;  								// OC3 active High

	TIM1->CCER &= ~BIT0;  								// CC1E  = 0
	TIM1->CCER &= ~BIT2;  								// CC1NE = 0
	TIM1->CCER &= ~BIT4;  								// CC2E  = 0
	TIM1->CCER &= ~BIT6;  								// CC2NE = 0
	TIM1->CCER &= ~BIT8;  								// CC3E  = 0
	TIM1->CCER &= ~BIT10; 								// CC3NE = 0

	TIM1->BDTR |= BIT15;  								// MOE =   1

	/* (11) 인터럽트 및 트리거 설정 */
	TIM1->DIER |= BIT0;					 				// Update 인터럽트 활성화
	TIM1->CR2  &= ~(BIT6 | BIT5 | BIT4);
	TIM1->CR2  |= BIT5;					 				// TRGO = Update (ADC 트리거)

	/*
	 * (12) 타이머 시작 (Center-aligned Mode 3)
	 * CMS 비트는 CCxIF (비교 인터럽트) 시점만 결정, Update Event에는 영향 없음
	 * 따라서 Center-aligned mode 3가 아닌 다른 mode로 정해도 무방함
	 * 모든 Center-aligned 모드에서 Update Event는 CNT=0과 CNT=ARR 양쪽에서 발생 (20kHz)
	 * RCR=1로 20kHz → 10kHz 분주 (main.c에서 설정)
	 */
	TIM1->CR1 |= (BIT6  | BIT5  						// Center-aligned mode 3
			   |  BIT2									// UG set으로 Update 인터럽트 발생하지 않게 하기
			   |  BIT0 );								// 카운터에 클럭 인가
}

/**
  * @brief  Disable selected PWM outputs
  * @param  U_DIS  Set to 1 to disable the U-phase PWM output
  * @param  V_DIS  Set to 1 to disable the V-phase PWM output
  * @param  W_DIS  Set to 1 to disable the W-phase PWM output
  * @retval None
  */
void Disable_UVW_PWM(uint8_t U_DIS, uint8_t V_DIS, uint8_t W_DIS)
{
	if(U_DIS == 1U)
	{
		TIM1->CCR1 = (uint16_t)(CNT_TOP); // Set channel 1(UTOP) PWM duty ratio to 0%
		TIM1->CCER &= ~BIT0;   	  		  // CC1E = 0, channel 1(UTOP) PWM disable(Low)
	}
	if(V_DIS == 1U)
	{
		TIM1->CCR2 = (uint16_t)(CNT_TOP); // Set channel 2(VTOP) PWM duty ratio to 0%
		TIM1->CCER &= ~BIT4;   	  		  // CC1E = 0, channel 2(VTOP) PWM disable(Low)
	}
	if(W_DIS == 1U)
	{
		TIM1->CCR3 = (uint16_t)(CNT_TOP); // Set channel 3(WTOP) PWM duty ratio to 0%
		TIM1->CCER &= ~BIT8;   	  		  // CC1E = 0, channel 3(WTOP) PWM disable(Low)
	}
}

/**
  * @brief  Enable selected PWM outputs with the specified duty ratios
  * @param  U_EN          Set to 1 to enable the U-phase PWM output
  * @param  V_EN          Set to 1 to enable the V-phase PWM output
  * @param  W_EN          Set to 1 to enable the W-phase PWM output
  * @param  U_Duty_Ratio  Duty ratio of the U-phase PWM output in percent
  * @param  V_Duty_Ratio  Duty ratio of the V-phase PWM output in percent
  * @param  W_Duty_Ratio  Duty ratio of the W-phase PWM output in percent
  * @retval None
  */
void Enable_UVW_PWM(uint8_t U_EN, uint8_t V_EN, uint8_t W_EN, float U_Duty_Ratio, float V_Duty_Ratio, float W_Duty_Ratio)
{
	/* Configure duty ratio */
	if(U_EN == 1U)
	{
		TIM1->CCR1 = (uint16_t)((1.0f-(U_Duty_Ratio/100.0f))*CNT_TOP); // channel 1(UTOP) PWM duty ratio 설정
	}
	if(V_EN == 1U)
	{
		TIM1->CCR2 = (uint16_t)((1.0f-(V_Duty_Ratio/100.0f))*CNT_TOP); // channel 2(VTOP) PWM duty ratio 설정
	}
	if(W_EN == 1U)
	{
		TIM1->CCR3 = (uint16_t)((1.0f-(W_Duty_Ratio/100.0f))*CNT_TOP); // channel 3(WTOP) PWM duty ratio 설정
	}

	/* Enable PWM */
	if(U_EN == 1U)
	{
		TIM1->CCER |= BIT0; // CC1E = 1, channel 1(UTOP) PWM enable
	}
	if(V_EN == 1U)
	{
		TIM1->CCER |= BIT4; // CC2E = 1, channel 2(VTOP) PWM enable
	}
	if(W_EN == 1U)
	{
		TIM1->CCER |= BIT8; // CC3E = 1, channel 3(WTOP) PWM enable
	}
}

/**
  * @brief  Initialize a periodic timer interrupt according to the required configuration
  * @param  MicroSec  Interrupt period in microseconds
  * @param  Priority  Interrupt priority
  * @retval None
  */
void Init_Timer_Intterupt_for_Control(uint16_t MicroSec, uint32_t Priority)
{
	/*
	 * @ Timer 용도
	 * 일정한 제어 주기 구현을 위한 Timer Interrupt 설정
	 *
	 * @ Timer 요구사항
	 * 1) 독립적인 Timer Interrupt
	 * 2) 모터 최대 속도 4.5 [rps] 기준으로 주파수 최솟값은 1.3kHz
	 *
	 * @ 사용하는 Pin
	 * None, TIM7 사용
	 */

	/* (1) TIM7 클럭 활성화 */
	RCC->APB1ENR |= BIT5;

    /* (2) Counter stop -> 안전장치 */
	TIM7->CR1 &= ~BIT0;

	/* (3) 카운터 기본 설정 */
	TIM7->PSC = 215u; 		  // 카운터 클럭 주파수 = 216MHz / (PSC+1) = 1MHz -> 1us 주기
	TIM7->ARR = (MicroSec-1); // timer interrupt 주기 = 1us x (ARR+1) = 700us -> 약 1.43kHz
	TIM7->EGR |= BIT0; 	  	  // UG(Update Generation) bit를 통한 Prescaler value 적용
	TIM7->CNT = 0u;   		  // counter 초기화

	/* (4) UIF(Update Interrupt Flag) clear -> 혹시 set 되어 있으면 clear */
	TIM7->SR &= ~BIT0;

    /* (5) Update interrupt enable */
    TIM7->DIER |= BIT0;

    /* (6) NVIC에서 TIM7 interrupt enable */
    NVIC_SetPriority(TIM7_IRQn, Priority);
    NVIC_EnableIRQ(TIM7_IRQn);

    /* (7) Counter start */
    TIM7->CR1 |= BIT0;
}

/**
  * @brief  Initialize a periodic timer interrupt according to the required configuration
  * @param  Microsec  Interrupt period in microseconds
  * @param  Priority  Interrupt priority
  * @retval None
  */
void Init_Timer_Intterupt_for_Timeout(uint32_t Microsec, uint32_t Priority)
{
	/*
	 * @ Timer 용도
	 * Timeout 구현
	 *
	 * @ Timer 요구사항
	 * 1) 독립적인 Timer Interrupt
	 * 2) 32 bit counter 내장
	 *
	 * @ 사용하는 Pin
	 * None, TIM5 사용
	 */

	/* (1) TIM5 클럭 활성화 */
	RCC->APB1ENR |= BIT3;

    /* (2) Counter stop -> 안전장치 */
	TIM5->CR1 &= ~BIT0;

	/* (3) 카운터 기본 설정 */
	TIM5->PSC = 215U; 	  	  // 카운터 클럭 주파수 = 216MHz / (PSC+1) = 1MHz -> 1us 주기
	TIM5->ARR = (Microsec-1); // timer interrupt 주기 = 1us x (ARR+1) = Microsec [us]
	TIM5->EGR |= BIT0; 	      // UG(Update Generation) bit를 통한 Prescaler value 적용
	TIM5->CNT = 0u;   		  // counter 초기화

	/* (4) UIF(Update Interrupt Flag) clear -> 혹시 set 되어 있으면 clear */
	TIM5->SR &= ~BIT0;

    /* (5) Update interrupt enable */
	TIM5->DIER |= BIT0;

    /* (6) NVIC에서 TIM3 interrupt enable */
    NVIC_SetPriority(TIM5_IRQn, Priority);
    NVIC_EnableIRQ(TIM5_IRQn);
}

/**
  * @brief  Start Timeout Counter
  * @param  None
  * @retval None
  */
void Start_Timeout(void)
{
	/* (1) Counter stop -> 안전장치 */
	TIM5->CR1 &= ~BIT0;

	/* (2) Counter 초기화 */
	TIM5->CNT = 0u;

	/* (3) UIF(Update Interrupt Flag) clear -> 혹시 set 되어 있으면 clear */
	TIM5->SR &= ~BIT0;

	/* (4) Counter start */
	TIM5->CR1 |= BIT0;
}

/**
  * @brief  Stop Timeout Counter
  * @param  None
  * @retval None
  */
void Stop_Timeout(void)
{
	/* (1) Counter stop */
	TIM5->CR1 &= ~BIT0;

	/* (2) Counter 초기화 */
	TIM5->CNT = 0u;

	/* (3) UIF(Update Interrupt Flag) clear -> 혹시 set 되어 있으면 clear */
	TIM5->SR &= ~BIT0;
}

/**
  * @brief  Initialize free running timer used for 1us counter
  * @param  None
  * @retval None
 */
void Init_Microsec_Timer(void)
{
	/*
	 * @ Timer 용도
	 * 시간 측정을 위한 1us 주기의 counter
	 *
	 * @ Timer 요구사항
	 * 1) counter 주기 1us
	 * 2) counter top은 max로 설정
	 *
	 * @ 사용하는 Pin
	 * None, TIM2 사용
	 */

	/* (1) TIM2 클럭 활성화 */
	RCC->APB1ENR |= BIT0;

    /* (2) Counter stop -> 안전장치 */
	TIM2->CR1 &= ~BIT0;

	/* (3) 카운터 기본 설정 */
	TIM2->PSC = 215U; 		  // 카운터 클럭 주파수 = 216MHz / (PSC+1) = 1MHz -> 1us 주기
	TIM2->ARR = 4294967295U;  // max로 설정
	TIM2->EGR |= BIT0; 	  	  // UG(Update Generation) bit를 통한 Prescaler value 적용
	TIM2->CNT = 0U;   		  // counter 초기화

	/* (4) UIF(Update Interrupt Flag) clear -> 혹시 set 되어 있으면 clear */
	TIM2->SR &= ~BIT0;

    /* (5) Counter start */
    TIM2->CR1 |= BIT0;
}

/**
  * @brief  Initialize the timer used for millisecond delay
  * @param  None
  * @retval None
 */
void Init_Delay_MilliSec(void)
{
	/*
	 * @ Timer 용도
	 * 정렬 후 대기 시간 구현
	 *
	 * @ Timer 요구사항
	 * 1) 최소 단위는 1ms
	 * 2) UIF(Update Interrupt Flag)를 polling 하여 시간 경과 확인
	 *
	 * @ 사용하는 Pin
	 * None, TIM6 사용
	 */

	/* (1) TIM6 클럭 활성화 */
	RCC->APB1ENR |= BIT4;

    /* (2) Counter stop -> 안전장치 */
	TIM6->CR1 &= ~BIT0;

	/* (3) 카운터 기본 설정 */
	TIM6->PSC = 215u; // 카운터 클럭 주파수 = 216MHz / (PSC+1) = 1MHz -> 1us 주기
	TIM6->ARR = 999u; // delay 시간 = 1us x (ARR+1) = 1000us -> 1ms
	TIM6->CNT = 0u;   // counter 초기화

	/* (4) Update event 발생 -> PSC, ARR 값을 실제 카운터에 반영 */
	TIM6->EGR |= BIT0;

	/* (5) UIF(Update Interrupt Flag) clear -> Update event로 인해 set 된 UIF clear */
	TIM6->SR &= ~BIT0;
}

/**
  * @brief  Generate a blocking delay for the specified time
  * @param  MilliSec  Delay time in milliseconds
  * @retval None
  */
void Delay_MilliSec(uint32_t MilliSec)
{
	for(int i = 0; i < MilliSec; i++)
	{
		/* (1) Counter stop -> 안전장치 */
		TIM6->CR1 &= ~BIT0;

		/* (2) Counter 초기화 */
		TIM6->CNT = 0u;

		/* (3) UIF(Update Interrupt Flag) clear -> 혹시 set 되어 있으면 clear */
		TIM6->SR &= ~BIT0;

		/* (4) Counter start */
		TIM6->CR1 |= BIT0;

		/* (5) UIF(Update Interrupt Flag)가 set 될 때까지 대기 */
		while((TIM6->SR & BIT0) == 0);

		/* (6) UIF(Update Interrupt Flag) clear */
		TIM6->SR &= ~BIT0;

		/* (7) Counter stop */
		TIM6->CR1 &= ~BIT0;

		/* (8) Counter 초기화 */
		TIM6->CNT = 0u;
	}
}


/* Private functions --------------------------------------------------------*/
/**
  * @brief
  * @param
  * @retval
  */
