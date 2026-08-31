
#include "stm32f767xx.h"
#include "clock.h"

void Init_Clock(void)
{
    //==========================================================================
    // (1) L1 캐시 활성화
    //==========================================================================
    SCB_EnableICache();
    SCB_EnableDCache();

    //==========================================================================
    // (2) Flash 설정: 7 wait states, ART 가속기, 프리페치
    //==========================================================================
    FLASH->ACR = FLASH_ACR_LATENCY_7WS | FLASH_ACR_ARTEN | FLASH_ACR_PRFTEN;

    //==========================================================================
    // (3) HSE/PLL 설정 → SYSCLK = 216MHz
    //==========================================================================
    RCC->CR |= RCC_CR_HSEON | RCC_CR_HSION;
    while ((RCC->CR & RCC_CR_HSIRDY) == 0);

    RCC->CFGR = 0;  // SYSCLK = HSI로 전환
    while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_HSI);

    RCC->CR = RCC_CR_HSEON | RCC_CR_HSION;  // PLL OFF 상태에서 설정

    // PLL: 16MHz * 216 / 8 / 2 = 216MHz
    RCC->PLLCFGR = RCC_PLLCFGR_PLLSRC_HSE
                 | (8U << RCC_PLLCFGR_PLLM_Pos)
                 | (216U << RCC_PLLCFGR_PLLN_Pos)
                 | (0U << RCC_PLLCFGR_PLLP_Pos)
                 | (9U << RCC_PLLCFGR_PLLQ_Pos);

    RCC->CR = RCC_CR_PLLON | RCC_CR_HSEON | RCC_CR_HSION;
    while ((RCC->CR & RCC_CR_PLLRDY) == 0);

    //==========================================================================
    // (4) Over-drive 모드 (216MHz 필수)
    //==========================================================================
    RCC->APB1ENR |= RCC_APB1ENR_PWREN;
    PWR->CR1 |= PWR_CR1_ODEN;
    while ((PWR->CSR1 & PWR_CSR1_ODRDY) == 0);
    PWR->CR1 |= PWR_CR1_ODSWEN;
    while ((PWR->CSR1 & PWR_CSR1_ODSWRDY) == 0);

    //==========================================================================
    // (5) 버스 클럭 분주: AHB=216MHz, APB1=APB2=54MHz
    //==========================================================================
    RCC->CFGR = RCC_CFGR_SW_PLL			  // PLL이 만든 clock을 SYSCLK로 설정
    									  // HCLK = SYSCLK / 1은 reset value
              | RCC_CFGR_PPRE1_DIV4		  // APB1 = HCLK / 4
              | RCC_CFGR_PPRE2_DIV4;	  // APB2 = HCLK / 4
    RCC->DCKCFGR1 = RCC_DCKCFGR1_TIMPRE;  // TIMx 클럭 = 216MHz
    while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL);
    RCC->CR |= RCC_CR_CSSON;

    //==========================================================================
    // (6) I/O 보상 셀 활성화
    //==========================================================================
    RCC->APB2ENR |= RCC_APB2ENR_SYSCFGEN;
    SYSCFG->CMPCR = SYSCFG_CMPCR_CMP_PD;
}


