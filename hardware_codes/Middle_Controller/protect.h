
#include <stdint.h>

void Init_Buzzer(void);
void Check_Safety_And_Alarm(uint8_t actuator_num, float *temp, float limit_temp, float *current, float *limit_current);