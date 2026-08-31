/**
  ******************************************************************************
  * @file    main.c
  * @author  Kwon Dohyeon
  * @brief   Position Control of BLDC Motor Using CAN.
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "stm32f767xx.h"
#include <math.h>
#include "register_macro.h"
#include "clock.h"
#include "gpio.h"
#include "timer.h"
#include "main.h"
#include "system_config.h"
#include "as5600_i2c.h"
#include "adc.h"
#include "control_block.h"
#include "motorCon.h"
#include "can_if.h"
#include "ak_servo.h"


/* Private typedef -----------------------------------------------------------*/
/* Interrupt Priority typedef */
typedef enum
{
	CURRENT_MEASURE_INT_PRIORITY = 1,
	SYSTIC_INT_PRIORITY,
	CONTROL_INT_PRIORITY,
	CAN_INT_PRIORITY,
	TIMEOUT_INT_PRIORITY,
} Int_Priority;


/* Private define ------------------------------------------------------------*/
/* Sampling Time of Control define */
#define TSAMP 700U // [us]

/* Timeout of Control define */
#define TIMEOUT (1U * 1000000U) // [us]

/* OP-AMP defines */
#define OPAMP_OFFSET 1.65F
#define OPAMP_GAIN   0.1647F


/* Private macros ------------------------------------------------------------*/
/*  */


/* Private variables ---------------------------------------------------------*/
/* 4015 BLDC Motor Handler variable */
BLDC_HandleTypeDef hbldc_4015;

/* MSBD Handler for Speed Caculation variable */
MSBD_HandleTypeDef hmsbd_SpeedCalc;

/* P Controller Handler for Position Control variable */
P_HandleTypeDef hp_PosCon;

/* PI Controller Handler for Speed Control variable */
PI_HandleTypeDef hpi_SpeedCon;

/* LPF Handler variable for Current */
LPF_HandleTypeDef hlpf_Ias;
LPF_HandleTypeDef hlpf_Ibs;
LPF_HandleTypeDef hlpf_Ics;

/* LPF Handler for Reference Smoothing variable */
LPF_HandleTypeDef hlpf_PosRef;

/* LPF Useage Switch variable */
uint8_t Using_LPF = 1U;
float PosRef_LPF = 0.0f;
float RPMRef = 0.0f;
float Degree_M_Abs = 0.0f;

/* Current Sensing Offset variavles */
float Ias_Offset = 0.0f;
float Ibs_Offset = 0.0f;
float Ics_Offset = 0.0f;

/* LED Toggle Period Count variable */
uint32_t FLT_LED_cnt = 0u;

/* AS5600 I2C Failure variables */
uint32_t AS5600_Try_Cnt = 0U;
uint32_t AS5600_Error_Cnt = 0U;
float Ratio_Of_Error = 0.0f;

/* CAN Timeout variables */
uint8_t Is_First_CAN_Error = 1U;
uint8_t Is_Timeout_Occur = 0U;


/* Private function prototypes -----------------------------------------------*/
/*  */


/* Private functions --------------------------------------------------------*/
/**
  * @brief  main function
  * @param  None
  * @retval int
  */
int main(void)
{
	/* Initialize Peripherals */
    MPU_Config();
    HAL_Init();
    Init_Clock();

    SystemCoreClock = 216000000U;

    if (HAL_InitTick(SYSTIC_INT_PRIORITY) != HAL_OK)
    {
        Error_Handler();
    }

    AS5600_I2C2_Init();

    Init_GPIO();
    Init_UVW_PWM();
    Init_ADC();
    Init_Timer_Intterupt_for_Timeout(TIMEOUT, TIMEOUT_INT_PRIORITY);
    Init_Microsec_Timer();
    CAN1_IF_Init();
    HAL_NVIC_SetPriority(CAN1_RX0_IRQn, CAN_INT_PRIORITY, 0);


    /* Configure 4015 BLDC Handler */
    hbldc_4015.Calibration_Offset = 292.0f;
    hbldc_4015.Align_Offset = 3.95f;
    hbldc_4015.PolePair = 11U;
    hbldc_4015.V_Limit = 12.0f;
    hbldc_4015.V_Inverter = 24.0f;
	BLDC_Init_State(&hbldc_4015);


	/* Configure MSBD Handler for Speed Caculation */
	hmsbd_SpeedCalc.Step_Num = 15U;
	MSBD_Init_State(&hmsbd_SpeedCalc);


	/* Configure PI Controller Handler for Speed Control */
	/* 무부하 */
//	hpi_SpeedCon.Kp = 0.15f;
//	hpi_SpeedCon.Ki = 15.0f;
	/* 부하 */
	hpi_SpeedCon.Kp = 1.5f;
	hpi_SpeedCon.Ki = 15.0f;

	hpi_SpeedCon.Ka = hpi_SpeedCon.Ki / hpi_SpeedCon.Kp;
	hpi_SpeedCon.OutMax = hbldc_4015.V_Limit;
	PI_Init_State(&hpi_SpeedCon);


	/* Configure P Controller Handler for Position Control */
	/* 무부하 */
//	hp_PosCon.Kp = 60.0f;
	/* 부하 */
	hp_PosCon.Kp = 60.0f;

	hp_PosCon.OutMax = 0.0f;


	/* Configure LPF Handler for Current*/
	hlpf_Ias.Fc = 50.0f;
	hlpf_Ibs.Fc = 50.0f;
	hlpf_Ics.Fc = 50.0f;
	hlpf_Ias.Tsamp = 50U;
	hlpf_Ibs.Tsamp = 50U;
	hlpf_Ics.Tsamp = 50U;
	LPF_Init_State(&hlpf_Ias);
	LPF_Init_State(&hlpf_Ibs);
	LPF_Init_State(&hlpf_Ics);
	LPF_Calc_Gain(&hlpf_Ias);
	LPF_Calc_Gain(&hlpf_Ibs);
	LPF_Calc_Gain(&hlpf_Ics);


	/* Configure LPF Handler for Reference Smoothing */
	// 3 Hz  -> very very smooth / little delay
	// 6 Hz  -> pretty smooth 	 / so little delay
	// 10 Hz -> little smooth 	 / so little delay
	// 15 Hz -> not  smooth 	 / almost no delay
	hlpf_PosRef.Fc = 6.0f;
	hlpf_PosRef.Tsamp = 700U;
	LPF_Init_State(&hlpf_PosRef);
	LPF_Calc_Gain(&hlpf_PosRef);


    /* Measure Current Offset (Average of 50 trials) */
    for (int i = 0; i < 50; i++)
    {
        ADC1->CR2 |= ADC_CR2_JSWSTART;
        while ((ADC->CSR & (ADC_CSR_JEOC1 | ADC_CSR_JEOC2 | ADC_CSR_JEOC3))
               != (ADC_CSR_JEOC1 | ADC_CSR_JEOC2 | ADC_CSR_JEOC3));

        Ias_Offset += ADC1->JDR1;
        Ibs_Offset += ADC2->JDR1;
        Ics_Offset += ADC3->JDR1;
    }

    Ias_Offset = (Ias_Offset / 50) - 2048;
    Ibs_Offset = (Ibs_Offset / 50) - 2048;
    Ics_Offset = (Ics_Offset / 50) - 2048;


    /* Initialize Current Measurement Interrupt */
    ADC1->CR2 |= ADC_CR2_JEXTEN_0;

    NVIC_SetPriority(TIM1_UP_TIM10_IRQn, CURRENT_MEASURE_INT_PRIORITY);
    NVIC_EnableIRQ(TIM1_UP_TIM10_IRQn);

    while(!(TIM1->CR1 & BIT4));
    TIM1->RCR = 1;


    /* Initialize Control Interrupt */
    Init_Timer_Intterupt_for_Control(TSAMP, CONTROL_INT_PRIORITY);


    while (1)
    {
    	/* The function internally sends feedback only once every 20 ms */
    	AK_Servo_SendFeedback_20ms(hbldc_4015.Degree_M, hbldc_4015.RPM_E, hbldc_4015.Idc);
    }
}

/**
  * @brief  Current Measurement Interrupt
  * @param  None
  * @retval None
  */
void TIM1_UP_TIM10_IRQHandler(void)
{
    if ((TIM1->SR & TIM_SR_UIF))
    {
        TIM1->SR &= ~TIM_SR_UIF;

    	/* PE14(Timing Verification Using Oscilloscope) High : Verified / duty ratio = 5% */
        GPIOE->BSRR = BIT14;

        /* Measure Current of 3 Phase */
        uint32_t result = 0;
        float Ias_tmp = 0.0f;
        float Ibs_tmp = 0.0f;
        float Ics_tmp = 0.0f;

        while (!(ADC->CSR & (ADC_CSR_JEOC1 | ADC_CSR_JEOC2 | ADC_CSR_JEOC3)));

        result = ADC1->JDR1;
        Ias_tmp = (-1.0f) * ((float)((float)result - (float)Ias_Offset) * ADC_VREF
                / (float)ADC_RESOLUTION - OPAMP_OFFSET)
                / OPAMP_GAIN;

        if(fabsf(Ias_tmp) > 10.0f)
        {
        	hbldc_4015.Ias = LPF_Calc(&hlpf_Ias, hbldc_4015.Ias);
        }
        else
        {
        	hbldc_4015.Ias = LPF_Calc(&hlpf_Ias, Ias_tmp);
        }

        result = ADC2->JDR1;
        Ibs_tmp =  (-1.0f) * ((float)((float)result - (float)Ibs_Offset) * ADC_VREF
                / (float)ADC_RESOLUTION - OPAMP_OFFSET)
                / OPAMP_GAIN;

        if(fabsf(Ibs_tmp) > 10.0f)
        {
        	hbldc_4015.Ibs = LPF_Calc(&hlpf_Ibs, hbldc_4015.Ibs);
        }
        else
        {
        	hbldc_4015.Ibs = LPF_Calc(&hlpf_Ibs, Ibs_tmp);
        }

        result = ADC3->JDR1;
        Ics_tmp =  (-1.0f) * ((float)((float)result - (float)Ics_Offset) * ADC_VREF
                / (float)ADC_RESOLUTION - OPAMP_OFFSET)
                / OPAMP_GAIN;

        if(fabsf(Ics_tmp) > 10.0f)
        {
        	hbldc_4015.Ics = LPF_Calc(&hlpf_Ics, hbldc_4015.Ics);
        }
        else
        {
        	hbldc_4015.Ics = LPF_Calc(&hlpf_Ics, Ics_tmp);
        }

        /* Current Protect */
        if(fabsf(hbldc_4015.Ias) > 4.5f || fabsf(hbldc_4015.Ibs) > 4.5f || fabsf(hbldc_4015.Ics) > 4.5f)
        {
        	hbldc_4015.OcCnt++;
            if(hbldc_4015.OcCnt>100)
            {
            	BLDC_PowerOff();
            	hbldc_4015.ErrCode = 1U;
            }
        }
        else
        {
        	hbldc_4015.OcCnt=0;
        }

        /* PE14(Timing Verification Using Oscilloscope) Low */
        GPIOE->BSRR = BIT30;
    }
}

/**
  * @brief  Control Interrupt
  * @param  None
  * @retval None
  */
void TIM7_IRQHandler(void)
{
    if (TIM7->SR & BIT0)
    {
    	/* Clear Interrupt Flag */
    	TIM7->SR &= ~BIT0;


    	/* PE14(Timing Verification Using Oscilloscope) High : Verified / duty ratio = 20% */
//        GPIOE->BSRR = BIT14;


        /* Read Reference Using CAN */
    	uint8_t Can_Rx_Result = AK_Servo_ProcessRx(hbldc_4015.PolePair);

        if(Can_Rx_Result != AK_CAN_ERROR)
        {
        	Stop_Timeout();
        	Is_Timeout_Occur = 0U;
        	Is_First_CAN_Error = 1U;

            /* PE14(Timing Verification Using Oscilloscope) Low : Verified */
//            GPIOE->BSRR = BIT30;
        }
        else
        {
        	if(Is_First_CAN_Error == 1U)
        	{
        		Start_Timeout();
        		Is_First_CAN_Error = 0U;

            	/* PE14(Timing Verification Using Oscilloscope) High */
//                GPIOE->BSRR = BIT14;
        	}
        }


        /* Position Control */
//        float Degree_M_Abs;
        uint32_t Current_Time;
        float PosRef;
        float SpeedRef;
        float VRef;

        uint16_t AS5600_Result = AS5600_ReadRawAngle();
        AS5600_Try_Cnt++;

        if(AS5600_Result != AS5600_READ_ERROR)
        {
			Degree_M_Abs = (-360.0f / (float)AS5600_RAW_ANGLE_MAX) * ((float)AS5600_Result) + 360.0f;
	        Current_Time = TIM2->CNT;

	        BLDC_Calc_Pos_N180_P180(&hbldc_4015, Degree_M_Abs);

	        BLDC_Calc_Sector(&hbldc_4015, Degree_M_Abs);

	        hbldc_4015.W_M = MSBD_Calc(&hmsbd_SpeedCalc, hbldc_4015.Theta, Current_Time);
	        hbldc_4015.RPM_M = hbldc_4015.W_M / (2.0f * PI) * 60.0f;
	        hbldc_4015.RPM_E = hbldc_4015.RPM_M * hbldc_4015.PolePair;

	        if((AK_Cmd_Count != 0U) && (Is_Timeout_Occur != 1U))
	        {
	        	PosRef_LPF = LPF_Calc(&hlpf_PosRef, AK_DegreeRef);

	        	if(Using_LPF == 1U)
	        	{
	        		PosRef = PosRef_LPF / 360.0f * (2.0f * PI);
	        	}
	        	else
	        	{
	        		PosRef = AK_DegreeRef / 360.0f * (2.0f * PI);
	        	}

	        	hp_PosCon.OutMax = AK_RPMmax / 60.0f * (2.0f * PI);
	        	SpeedRef = P_Calc(&hp_PosCon, PosRef, hbldc_4015.Theta);

	        	RPMRef = SpeedRef / (2.0f * PI) * 60.0f;

	        	hpi_SpeedCon.Ka = hpi_SpeedCon.Ki / hpi_SpeedCon.Kp;
	        	VRef = PI_Calc(&hpi_SpeedCon, SpeedRef, hbldc_4015.W_M, Current_Time);

	        	BLDC_6Step_Commutation_CCW(&hbldc_4015, VRef);
	        }
	        else
	        {
	        	BLDC_PowerOff();
	        	MSBD_Init_State(&hmsbd_SpeedCalc);
	        	PI_Init_State(&hpi_SpeedCon);
	        	LPF_Init_State(&hlpf_PosRef);
	        }
        }
        else
        {
        	BLDC_PowerOff();
        	MSBD_Init_State(&hmsbd_SpeedCalc);
        	PI_Init_State(&hpi_SpeedCon);
        	LPF_Init_State(&hlpf_PosRef);

        	AS5600_Error_Cnt++;
        }

        Ratio_Of_Error = (float)AS5600_Error_Cnt / (float)AS5600_Try_Cnt;


        /* LED Toggle Every 0.7s */
        if ((FLT_LED_cnt % 1000u) == 0u)
        {
        	GPIOC->ODR ^= BIT6;
        }

        FLT_LED_cnt++;


        /* PE14(Timing Verification Using Oscilloscope) Low */
//        GPIOE->BSRR = BIT30;
    }
}

/**
  * @brief  Timeout Interrupt
  * @param  None
  * @retval None
  */
void TIM5_IRQHandler(void)
{
	if (TIM5->SR & BIT0)
	{
    	/* Clear Interrupt Flag */
		TIM5->SR &= ~BIT0;

    	/* Set Timeout Flag */
    	Is_Timeout_Occur = 1U;

        /* PE14(Timing Verification Using Oscilloscope) Low */
//        GPIOE->BSRR = BIT30;
	}
}

/**
  * @brief  This function is executed in case of error occurrence.
  * @param  None
  * @retval None
  */
void Error_Handler(void)
{
    __disable_irq();

    while (1)
    {

    }
}
