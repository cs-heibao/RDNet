
class logg():
    @staticmethod
    def logger(logger, loss, level_loss, glob_step):
        logger.add_scalar('total loss', loss, global_step=glob_step)
        # ========== level_16 ============
        logger.add_scalar('level_16/xy_loss', float(level_loss[0].split(',')[0].split(' ')[-1]),
                          global_step=glob_step)
        logger.add_scalar('level_16/xy_loss', float(level_loss[0].split(',')[1].split(' ')[-1]),
                          global_step=glob_step)
        logger.add_scalar('level_16/xy_loss', float(level_loss[0].split(',')[2].split(' ')[-1]),
                          global_step=glob_step)
        logger.add_scalar('level_16/xy_loss', float(level_loss[0].split(',')[3].split(' ')[-1]),
                          global_step=glob_step)

        # ========== level_32 ============
        logger.add_scalar('level_32/xy_loss', float(level_loss[1].split(',')[0].split(' ')[-1]),
                          global_step=glob_step)
        logger.add_scalar('level_32/xy_loss', float(level_loss[1].split(',')[1].split(' ')[-1]),
                          global_step=glob_step)
        logger.add_scalar('level_32/xy_loss', float(level_loss[1].split(',')[2].split(' ')[-1]),
                          global_step=glob_step)
        logger.add_scalar('level_32/xy_loss', float(level_loss[1].split(',')[3].split(' ')[-1]),
                          global_step=glob_step)

        # ========== level_64 ============
        logger.add_scalar('level_64/xy_loss', float(level_loss[2].split(',')[0].split(' ')[-1]),
                          global_step=glob_step)
        logger.add_scalar('level_64/xy_loss', float(level_loss[2].split(',')[1].split(' ')[-1]),
                          global_step=glob_step)
        logger.add_scalar('level_64/xy_loss', float(level_loss[2].split(',')[2].split(' ')[-1]),
                          global_step=glob_step)
        logger.add_scalar('level_64/xy_loss', float(level_loss[2].split(',')[3].split(' ')[-1]),
                          global_step=glob_step)
