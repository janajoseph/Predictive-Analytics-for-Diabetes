/*
SQLyog Enterprise - MySQL GUI v6.56
MySQL - 5.5.5-10.1.13-MariaDB : Database - diabetes
*********************************************************************
*/


/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;

CREATE DATABASE /*!32312 IF NOT EXISTS*/`diabetes` /*!40100 DEFAULT CHARACTER SET latin1 */;

USE `diabetes`;

/*Table structure for table `dc` */

DROP TABLE IF EXISTS `dc`;

CREATE TABLE `dc` (
  `id` int(200) NOT NULL AUTO_INCREMENT,
  `Email` varchar(200) DEFAULT NULL,
  `Password` varchar(200) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=latin1;

/*Data for the table `dc` */

insert  into `dc`(`id`,`Email`,`Password`) values (1,'janajoseph@gmail.com','Chill'),(2,'cse.takeoff@gmail.com','23122002');
