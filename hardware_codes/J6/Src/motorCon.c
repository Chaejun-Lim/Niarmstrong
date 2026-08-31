/**
  ******************************************************************************
  * @file    motorCon.c
  * @author  Kwon Dohyeon
  * @brief   BLDC Motor Control module.
  *          This file provides functions to Control BLDC Motor.
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "stm32f767xx.h"
#include "gpio.h"
#include "timer.h"
#include "control_block.h"
#include <math.h>
#include "motorCon.h"


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
  * @brief  Initialize all states of the BLDC motor
  * @param  hbldc Pointer to a BLDC_HandleTypeDef structure that contains
  *               the configuration and states of the BLDC motor
  * @retval None
  */
void BLDC_Init_State(BLDC_HandleTypeDef *hbldc)
{
	hbldc->Degree_M = 0.0f;
	hbldc->Theta    = 0.0f;
	hbldc->W_M      = 0.0f;
	hbldc->RPM_M    = 0.0f;
	hbldc->RPM_E    = 0.0f;
	hbldc->Sector   = 0U;
	hbldc->Sector_p = 0U;
	hbldc->rule     = 0U;
	hbldc->Ias		= 0.0f;
	hbldc->Ibs		= 0.0f;
	hbldc->Ics		= 0.0f;
	hbldc->Idc      = 0.0f;
	hbldc->OcCnt    = 0U;
	hbldc->ErrCode  = 0U;
}

/**
  * @brief  Initialize GPIO peripherals according to the required configuration
  * @param  None
  * @retval None
  */
void BLDC_Align(void)
{
	/*
	 * Function for aligning the rotor of BLDC motor to the center of Sector 6
	 *
	 * Configure current to flow from phases V and W to phase U
	 * UTOP : OFF 	VTOP : 30% 	  WTOP : 30%
	 * UBOT : ON	VBOT : OFF    WBOT : OFF
	 */

	/* Turn Off Unused Phases */
	Disable_UVW_PWM(1, 0, 0);
	Reset_UVW_GPIO(0, 1, 1);

	/* Turn On Active Phases */
	Enable_UVW_PWM(0, 1, 1, 0.0f, 30.0f, 30.0f);
	Set_UVW_GPIO(1, 0, 0);
}

/**
  * @brief  Make all FET Low
  * @param  None
  * @retval None
  */
void BLDC_PowerOff(void)
{
	/* Force All Outputs Low */
	Disable_UVW_PWM(1, 1, 1);
	Reset_UVW_GPIO(1, 1, 1);
}

/**
  * @brief  Calculate the calibrated mechanical position in the range of
  *         -180 to 180 degrees and convert it to radians
  * @param  hbldc        Pointer to a BLDC_HandleTypeDef structure that contains
  *                      the configuration and states of the BLDC motor
  * @param  Degree_M_Abs Absolute mechanical angle in degrees
  * @retval None
  */
void BLDC_Calc_Pos_N180_P180(BLDC_HandleTypeDef *hbldc, float Degree_M_Abs)
{
	/* declaration of local variables */
	float Calibrated_Degree_M_Abs;

	/* Apply Calibration Offset */
	Calibrated_Degree_M_Abs = Degree_M_Abs - hbldc->Calibration_Offset;

	/* Calculate mechanical angle in range of -180 ~ 180 [degree] */
	if	   (Calibrated_Degree_M_Abs >  180.0f) {hbldc->Degree_M = Calibrated_Degree_M_Abs - 360.0f;}
	else if(Calibrated_Degree_M_Abs < -180.0f) {hbldc->Degree_M = Calibrated_Degree_M_Abs + 360.0f;}
	else						       		   {hbldc->Degree_M = Calibrated_Degree_M_Abs;		   }

	/* [degree] -> [rad] */
	hbldc->Theta = hbldc->Degree_M / 360.0f * (2 * PI);
}

/**
  * @brief  Calculate the electrical angle and determine the commutation sector
  * @param  hbldc        Pointer to a BLDC_HandleTypeDef structure that contains
  *                      the configuration and states of the BLDC motor
  * @param  Degree_M_Abs Absolute mechanical angle in degrees
  * @retval None
  */
void BLDC_Calc_Sector(BLDC_HandleTypeDef *hbldc, float Degree_M_Abs)
{
	/* declaration of local variables */
	float Aligned_Degree_M_Abs, Degree_E_Abs;

	/* Apply Align Offset */
	Aligned_Degree_M_Abs = Degree_M_Abs - hbldc->Align_Offset;

	/* Calculate mechanical angle in range of 0 ~ 360 [degree] */
	if(Aligned_Degree_M_Abs < 0.0f)
	{
		Aligned_Degree_M_Abs += 360.0f;
	}

	/* Calculate electrical angle in range of 0 ~ 360 [E degree] */
	Degree_E_Abs = fmodf((Aligned_Degree_M_Abs * (float)(hbldc->PolePair)), 360.0f);

	/* Select Sector */
	if	   ((Degree_E_Abs >=  30.0f) && (Degree_E_Abs <  90.0f)) {hbldc->Sector = 1u;}
	else if((Degree_E_Abs >=  90.0f) && (Degree_E_Abs < 150.0f)) {hbldc->Sector = 2u;}
	else if((Degree_E_Abs >= 150.0f) && (Degree_E_Abs < 210.0f)) {hbldc->Sector = 3u;}
	else if((Degree_E_Abs >= 210.0f) && (Degree_E_Abs < 270.0f)) {hbldc->Sector = 4u;}
	else if((Degree_E_Abs >= 270.0f) && (Degree_E_Abs < 330.0f)) {hbldc->Sector = 5u;}
	else if((Degree_E_Abs >= 330.0f) || (Degree_E_Abs <  30.0f)) {hbldc->Sector = 6u;}
	else 													   	 {hbldc->Sector = 0u;}
}

/**
  * @brief  Measure and update the motor current variables.
  * @param  hbldc  Pointer to the BLDC handle structure
  * @retval None
  */
void BLDC_Calc_DC_Current(BLDC_HandleTypeDef *hbldc)
{
	/* declaration of local variables */
	int8_t Sector_diff;

	/* Calculate Equivalent DC Current */
	Sector_diff = (int8_t)(hbldc->Sector) - (int8_t)(hbldc->Sector_p);
	hbldc->Sector_p = hbldc->Sector;

	if	   (Sector_diff >  3) {Sector_diff -= 6;}
	else if(Sector_diff < -3) {Sector_diff += 6;}
	else					  {}

	if	   (Sector_diff >  0) {hbldc->rule = 1u;}
	else if(Sector_diff <  0) {hbldc->rule = 2u;}
	else					  {}

	if 	   (hbldc->rule == 1)
	{
		if	   (hbldc->Sector == 1) {hbldc->Idc = -hbldc->Ibs;}
		else if(hbldc->Sector == 2) {hbldc->Idc =  hbldc->Ias;}
		else if(hbldc->Sector == 3) {hbldc->Idc = -hbldc->Ics;}
		else if(hbldc->Sector == 4) {hbldc->Idc =  hbldc->Ibs;}
		else if(hbldc->Sector == 5) {hbldc->Idc = -hbldc->Ias;}
		else if(hbldc->Sector == 6) {hbldc->Idc =  hbldc->Ics;}
		else 				 		{hbldc->Idc = 0.0f;}
	}
	else if(hbldc->rule == 2)
	{
		if	   (hbldc->Sector == 1) {hbldc->Idc =  hbldc->Ias;}
		else if(hbldc->Sector == 2) {hbldc->Idc = -hbldc->Ics;}
		else if(hbldc->Sector == 3) {hbldc->Idc =  hbldc->Ibs;}
		else if(hbldc->Sector == 4) {hbldc->Idc = -hbldc->Ias;}
		else if(hbldc->Sector == 5) {hbldc->Idc =  hbldc->Ics;}
		else if(hbldc->Sector == 6) {hbldc->Idc = -hbldc->Ibs;}
		else 				 		{hbldc->Idc = 0.0f;}
	}
	else					 		{hbldc->Idc = 0.0f;}
}

/**
  * @brief  Apply the reference voltage using 6-step commutation when UVW Phases are aligned CCW Order
  * @param  VRef  Reference voltage for determining the PWM duty ratio
  * 			  and commutation direction
  * @retval None
  */
void BLDC_6Step_Commutation_CCW(BLDC_HandleTypeDef *hbldc, float VRef)
{
	/* Protect */
	if(hbldc->ErrCode == 1u)
	{
		BLDC_PowerOff();
		return;
	}

	/* Calculate PWM duty ratio */
	VRef = MC_SAT(VRef, hbldc->V_Limit);
	float PWM_duty_ratio = VRef / hbldc->V_Inverter * 100.0f;

	/* 6-step commutation */
	if(PWM_duty_ratio >= 0.0f)
	{
		if(hbldc->Sector == 1u) 	 /* U->V */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(0, 1, 1);
			Reset_UVW_GPIO(1, 0, 1);

			/* Turn on active phases */
			Enable_UVW_PWM(1, 0, 0, PWM_duty_ratio, 0.0f, 0.0f);
			Set_UVW_GPIO(0, 1, 0);
		}
		else if(hbldc->Sector == 2u) /* U->W */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(0, 1, 1);
			Reset_UVW_GPIO(1, 1, 0);

			/* Turn on active phases */
			Enable_UVW_PWM(1, 0, 0, PWM_duty_ratio, 0.0f, 0.0f);
			Set_UVW_GPIO(0, 0, 1);
		}
		else if(hbldc->Sector == 3u) /* V->W */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(1, 0, 1);
			Reset_UVW_GPIO(1, 1, 0);

			/* Turn on active phases */
			Enable_UVW_PWM(0, 1, 0, 0.0f, PWM_duty_ratio, 0.0f);
			Set_UVW_GPIO(0, 0, 1);
		}
		else if(hbldc->Sector == 4u) /* V->U */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(1, 0, 1);
			Reset_UVW_GPIO(0, 1, 1);

			/* Turn on active phases */
			Enable_UVW_PWM(0, 1, 0, 0.0f, PWM_duty_ratio, 0.0f);
			Set_UVW_GPIO(1, 0, 0);
		}
		else if(hbldc->Sector == 5u) /* W->U */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(1, 1, 0);
			Reset_UVW_GPIO(0, 1, 1);

			/* Turn on active phases */
			Enable_UVW_PWM(0, 0, 1, 0.0f, 0.0f, PWM_duty_ratio);
			Set_UVW_GPIO(1, 0, 0);
		}
		else if(hbldc->Sector == 6u) /* W->V */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(1, 1, 0);
			Reset_UVW_GPIO(1, 0, 1);

			/* Turn on active phases */
			Enable_UVW_PWM(0, 0, 1, 0.0f, 0.0f, PWM_duty_ratio);
			Set_UVW_GPIO(0, 1, 0);
		}
		else /* When Sector is not in the range of 1 ~ 6 */
		{
			/* Force all outputs low */
			BLDC_PowerOff();
		}
	}
	else
	{
		/* only the magnitude is used for the PWM duty ratio */
		PWM_duty_ratio = -PWM_duty_ratio;

		if(hbldc->Sector == 1u) 	 /* V->U */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(1, 0, 1);
			Reset_UVW_GPIO(0, 1, 1);

			/* Turn on active phases */
			Enable_UVW_PWM(0, 1, 0, 0.0f, PWM_duty_ratio, 0.0f);
			Set_UVW_GPIO(1, 0, 0);
		}
		else if(hbldc->Sector == 2u) /* W->U */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(1, 1, 0);
			Reset_UVW_GPIO(0, 1, 1);

			/* Turn on active phases */
			Enable_UVW_PWM(0, 0, 1, 0.0f, 0.0f, PWM_duty_ratio);
			Set_UVW_GPIO(1, 0, 0);
		}
		else if(hbldc->Sector == 3u) /* W->V */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(1, 1, 0);
			Reset_UVW_GPIO(1, 0, 1);

			/* Turn on active phases */
			Enable_UVW_PWM(0, 0, 1, 0.0f, 0.0f, PWM_duty_ratio);
			Set_UVW_GPIO(0, 1, 0);
		}
		else if(hbldc->Sector == 4u) /* U->V */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(0, 1, 1);
			Reset_UVW_GPIO(1, 0, 1);

			/* Turn on active phases */
			Enable_UVW_PWM(1, 0, 0, PWM_duty_ratio, 0.0f, 0.0f);
			Set_UVW_GPIO(0, 1, 0);
		}
		else if(hbldc->Sector == 5u) /* U->W */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(0, 1, 1);
			Reset_UVW_GPIO(1, 1, 0);

			/* Turn on active phases */
			Enable_UVW_PWM(1, 0, 0, PWM_duty_ratio, 0.0f, 0.0f);
			Set_UVW_GPIO(0, 0, 1);
		}
		else if(hbldc->Sector == 6u) /* V->W */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(1, 0, 1);
			Reset_UVW_GPIO(1, 1, 0);

			/* Turn on active phases */
			Enable_UVW_PWM(0, 1, 0, 0.0f, PWM_duty_ratio, 0.0f);
			Set_UVW_GPIO(0, 0, 1);
		}
		else /* When Sector is not in the range of 1 ~ 6 */
		{
			/* Force all outputs low */
			BLDC_PowerOff();
		}
	}
}

/**
  * @brief  Apply the reference voltage using 6-step commutation when UVW Phases are aligned CW Order
  * @param  VRef  Reference voltage for determining the PWM duty ratio
  * 			  and commutation direction
  * @retval None
  */
void BLDC_6Step_Commutation_CW(BLDC_HandleTypeDef *hbldc, float VRef)
{
	/* Protect */
	if(hbldc->ErrCode == 1u)
	{
		BLDC_PowerOff();
		return;
	}

	/* Calculate PWM duty ratio */
	VRef = MC_SAT(VRef, hbldc->V_Limit);
	float PWM_duty_ratio = VRef / hbldc->V_Inverter * 100.0f;

	/* 6-step commutation */
	if(PWM_duty_ratio >= 0.0f)
	{
		if(hbldc->Sector == 1u) 	 /* U->W */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(0, 1, 1);
			Reset_UVW_GPIO(1, 1, 0);

			/* Turn on active phases */
			Enable_UVW_PWM(1, 0, 0, PWM_duty_ratio, 0.0f, 0.0f);
			Set_UVW_GPIO(0, 0, 1);
		}
		else if(hbldc->Sector == 2u) /* U->V */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(0, 1, 1);
			Reset_UVW_GPIO(1, 0, 1);

			/* Turn on active phases */
			Enable_UVW_PWM(1, 0, 0, PWM_duty_ratio, 0.0f, 0.0f);
			Set_UVW_GPIO(0, 1, 0);
		}
		else if(hbldc->Sector == 3u) /* W->V */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(1, 1, 0);
			Reset_UVW_GPIO(1, 0, 1);

			/* Turn on active phases */
			Enable_UVW_PWM(0, 0, 1, 0.0f, 0.0f, PWM_duty_ratio);
			Set_UVW_GPIO(0, 1, 0);
		}
		else if(hbldc->Sector == 4u) /* W->U */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(1, 1, 0);
			Reset_UVW_GPIO(0, 1, 1);

			/* Turn on active phases */
			Enable_UVW_PWM(0, 0, 1, 0.0f, 0.0f, PWM_duty_ratio);
			Set_UVW_GPIO(1, 0, 0);
		}
		else if(hbldc->Sector == 5u) /* V->U */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(1, 0, 1);
			Reset_UVW_GPIO(0, 1, 1);

			/* Turn on active phases */
			Enable_UVW_PWM(0, 1, 0, 0.0f, PWM_duty_ratio, 0.0f);
			Set_UVW_GPIO(1, 0, 0);
		}
		else if(hbldc->Sector == 6u) /* V->W */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(1, 0, 1);
			Reset_UVW_GPIO(1, 1, 0);

			/* Turn on active phases */
			Enable_UVW_PWM(0, 1, 0, 0.0f, PWM_duty_ratio, 0.0f);
			Set_UVW_GPIO(0, 0, 1);
		}
		else /* When Sector is not in the range of 1 ~ 6 */
		{
			/* Force all outputs low */
			BLDC_PowerOff();
		}
	}
	else
	{
		/* only the magnitude is used for the PWM duty ratio */
		PWM_duty_ratio = -PWM_duty_ratio;

		if(hbldc->Sector == 1u) 	 /* W->U */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(1, 1, 0);
			Reset_UVW_GPIO(0, 1, 1);

			/* Turn on active phases */
			Enable_UVW_PWM(0, 0, 1, 0.0f, 0.0f, PWM_duty_ratio);
			Set_UVW_GPIO(1, 0, 0);
		}
		else if(hbldc->Sector == 2u) /* V->U */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(1, 0, 1);
			Reset_UVW_GPIO(0, 1, 1);

			/* Turn on active phases */
			Enable_UVW_PWM(0, 1, 0, 0.0f, PWM_duty_ratio, 0.0f);
			Set_UVW_GPIO(1, 0, 0);
		}
		else if(hbldc->Sector == 3u) /* V->W */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(1, 0, 1);
			Reset_UVW_GPIO(1, 1, 0);

			/* Turn on active phases */
			Enable_UVW_PWM(0, 1, 0, 0.0f, PWM_duty_ratio, 0.0f);
			Set_UVW_GPIO(0, 0, 1);
		}
		else if(hbldc->Sector == 4u) /* U->W */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(0, 1, 1);
			Reset_UVW_GPIO(1, 1, 0);

			/* Turn on active phases */
			Enable_UVW_PWM(1, 0, 0, PWM_duty_ratio, 0.0f, 0.0f);
			Set_UVW_GPIO(0, 0, 1);
		}
		else if(hbldc->Sector == 5u) /* U->V */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(0, 1, 1);
			Reset_UVW_GPIO(1, 0, 1);

			/* Turn on active phases */
			Enable_UVW_PWM(1, 0, 0, PWM_duty_ratio, 0.0f, 0.0f);
			Set_UVW_GPIO(0, 1, 0);
		}
		else if(hbldc->Sector == 6u) /* W->V */
		{
			/* Turn off unused phases */
			Disable_UVW_PWM(1, 1, 0);
			Reset_UVW_GPIO(1, 0, 1);

			/* Turn on active phases */
			Enable_UVW_PWM(0, 0, 1, 0.0f, 0.0f, PWM_duty_ratio);
			Set_UVW_GPIO(0, 1, 0);
		}
		else /* When Sector is not in the range of 1 ~ 6 */
		{
			/* Force all outputs low */
			BLDC_PowerOff();
		}
	}
}


/* Private functions --------------------------------------------------------*/
/**
  * @brief
  * @param
  * @retval
  */
