import torch

# data
features = "Ljava/net/URL;->openConnection,Ljava/net/URL;->openStream,Ljava/net/URL;->getContent," \
           "Landroid/telephony/TelephonyManager;->getCallState,Landroid/telephony/TelephonyManager;->getCellLocation," \
           "Landroid/telephony/TelephonyManager;->getDeviceId," \
           "Landroid/telephony/TelephonyManager;->getDeviceSoftwareVersion," \
           "Landroid/telephony/TelephonyManager;->getNeighboringCellInfo," \
           "Landroid/telephony/TelephonyManager;->getNetworkCountryIso," \
           "Landroid/telephony/TelephonyManager;->getNetworkOperator," \
           "Landroid/telephony/TelephonyManager;->getNetworkOperatorName," \
           "Landroid/telephony/TelephonyManager;->getNetworkType,Landroid/telephony/TelephonyManager;->getPhoneType," \
           "Landroid/telephony/TelephonyManager;->getSimCountryIso," \
           "Landroid/telephony/TelephonyManager;->getSimOperator," \
           "Landroid/telephony/TelephonyManager;->getSimOperatorName," \
           "Landroid/telephony/TelephonyManager;->getSimSerialNumber," \
           "Landroid/telephony/TelephonyManager;->getSimState,Landroid/telephony/TelephonyManager;->getSubscriberId," \
           "Landroid/telephony/TelephonyManager;->getVoiceMailNumber," \
           "Landroid/telephony/TelephonyManager;->hasIccCard,Landroid/telephony/TelephonyManager;->isNetworkRoaming," \
           "Landroid/telephony/gsm/SmsManager;->divideMessage,Landroid/telephony/gsm/SmsManager;->getDefault," \
           "Landroid/telephony/gsm/SmsManager;->sendDataMessage," \
           "Landroid/telephony/gsm/SmsManager;->sendMultipartTextMessage," \
           "Landroid/telephony/gsm/SmsManager;->sendTextMessage,Ljava/net/HttpURLConnection;->disconnect," \
           "Ljava/net/HttpURLConnection;->getContentEncoding,Ljava/net/HttpURLConnection;->getPermission," \
           "Ljava/net/HttpURLConnection;->getRequestMethod,Ljava/net/HttpURLConnection;->getResponseCode," \
           "Ljava/net/HttpURLConnection;->getResponseMessage,Ljava/net/HttpURLConnection;->usingProxy," \
           "Landroid/content/ContentResolver;->bulkInsert,Landroid/content/ContentResolver;->getType," \
           "Landroid/content/ContentResolver;->openAssetFileDescriptor,Landroid/content/ContentResolver;->query," \
           "Landroid/content/ContentResolver;->registerContentObserver,Landroid/content/ContentResolver;->update," \
           "Landroid/content/ContentResolver;->delete,Ljava/lang/Runtime;->getRuntime,Ljava/lang/Runtime;->exec," \
           "Ljava/lang/Runtime;->addShutdownHook,Ljava/lang/Runtime;->maxMemory," \
           "Ljava/net/URLConnection;->addRequestProperty,Ljava/net/URLConnection;->connect," \
           "Ljava/net/URLConnection;->getConnectTimeout,Ljava/net/URLConnection;->getContent," \
           "Ljava/net/URLConnection;->getContentType,Ljava/net/URLConnection;->getDefaultUseCaches," \
           "Ljava/net/URLConnection;->getPermission,Ljava/net/URLConnection;->getURL," \
           "Ljava/net/URLConnection;->setConnectTimeout,Ljava/net/URLConnection;->setReadTimeout," \
           "Landroid/app/ActivityManager;->getLargeMemoryClass,Landroid/app/ActivityManager;->getRunningAppProcesses," \
           "Landroid/app/ActivityManager;->isLowRamDevice,Landroid/app/ActivityManager;->killBackgroundProcesses," \
           "Landroid/app/ActivityManager;->restartPackage,Landroid/bluetooth/BluetoothAdapter;->cancelDiscovery," \
           "Landroid/bluetooth/BluetoothAdapter;->getAddress,Landroid/bluetooth/BluetoothAdapter;->getBondedDevices," \
           "Landroid/bluetooth/BluetoothAdapter;->getRemoteDevice,Landroid/bluetooth/BluetoothAdapter;->connect," \
           "Landroid/app/DownloadManager;->enqueue,Landroid/app/DownloadManager;->query," \
           "Landroid/location/LocationManager;->addGpsStatusListener," \
           "Landroid/location/LocationManager;->addNmeaListener," \
           "Landroid/location/LocationManager;->addProximityAlert," \
           "Landroid/location/LocationManager;->addTestProvider," \
           "Landroid/location/LocationManager;->clearTestProviderLocation," \
           "Landroid/location/LocationManager;->getBestProvider,Landroid/location/LocationManager;->getGpsStatus," \
           "Landroid/location/LocationManager;->getLastKnownLocation," \
           "Landroid/location/LocationManager;->requestLocationUpdates," \
           "Landroid/location/LocationManager;->sendExtraCommand,Landroid/net/wifi/WifiManager;->addNetwork," \
           "Landroid/net/wifi/WifiManager;->calculateSignalLevel,Landroid/net/wifi/WifiManager;->createWifiLock," \
           "Landroid/net/wifi/WifiManager;->disconnect,Landroid/net/wifi/WifiManager;->enableNetwork," \
           "Landroid/net/wifi/WifiManager;->getConfiguredNetworks,Landroid/net/wifi/WifiManager;->getConnectionInfo," \
           "Landroid/net/wifi/WifiManager;->getDhcpInfo,Landroid/net/wifi/WifiManager;->getScanResults," \
           "Landroid/net/wifi/WifiManager;->getWifiState,Landroid/net/wifi/WifiManager;->isWifiEnabled," \
           "Landroid/net/wifi/WifiManager;->removeNetwork,Landroid/net/wifi/WifiManager;->saveConfiguration," \
           "Landroid/net/wifi/WifiManager;->setWifiEnabled,Landroid/app/NotificationManager;->cancel," \
           "Landroid/app/NotificationManager;->notify,Landroid/content/pm/PackageManager;->checkPermission," \
           "Landroid/os/PowerManager;->isInteractive,Landroid/os/PowerManager;->isScreenOn," \
           "Landroid/os/PowerManager;->newWakeLock,android.permission.ACCESS_COARSE_LOCATION," \
           "android.permission.ACCESS_FINE_LOCATION,android.permission.ACCESS_LOCATION_EXTRA_COMMANDS," \
           "android.permission.ACCESS_NETWORK_STATE,android.permission.ACCESS_WIFI_STATE," \
           "android.permission.AUTHENTICATE_ACCOUNTS,android.permission.BATTERY_STATS,android.permission.BLUETOOTH," \
           "android.permission.BROADCAST_SMS,android.permission.BROADCAST_STICKY,android.permission.CALL_PHONE," \
           "android.permission.CAMERA,android.permission.CHANGE_COMPONENT_ENABLED_STATE," \
           "android.permission.CHANGE_CONFIGURATION,android.permission.CHANGE_NETWORK_STATE," \
           "android.permission.CHANGE_WIFI_MULTICAST_STATE,android.permission.CHANGE_WIFI_STATE," \
           "android.permission.CLEAR_APP_CACHE,android.permission.CONTROL_LOCATION_UPDATES," \
           "android.permission.DELETE_CACHE_FILES,android.permission.DELETE_PACKAGES,android.permission.DEVICE_POWER," \
           "android.permission.DISABLE_KEYGUARD,android.permission.EXPAND_STATUS_BAR,android.permission.FLASHLIGHT," \
           "android.permission.GET_PACKAGE_SIZE,android.permission.GET_TASKS,android.permission.INSTALL_PACKAGES," \
           "android.permission.INTERNET,android.permission.KILL_BACKGROUND_PROCESSES," \
           "android.permission.MODIFY_PHONE_STATE,android.permission.MOUNT_UNMOUNT_FILESYSTEMS," \
           "android.permission.NFC,android.permission.PERSISTENT_ACTIVITY,android.permission.PROCESS_OUTGOING_CALLS," \
           "android.permission.READ_CALL_LOG,android.permission.READ_CONTACTS," \
           "android.permission.READ_EXTERNAL_STORAGE,android.permission.READ_LOGS," \
           "android.permission.READ_PHONE_STATE,android.permission.READ_PROFILE,android.permission.READ_SMS," \
           "android.permission.RECEIVE_BOOT_COMPLETED,android.permission.RECEIVE_MMS,android.permission.RECEIVE_SMS," \
           "android.permission.RECEIVE_WAP_PUSH,android.permission.RECORD_AUDIO,android.permission.RESTART_PACKAGES," \
           "android.permission.SEND_SMS,android.permission.SET_WALLPAPER,android.permission.SET_WALLPAPER_HINTS," \
           "android.permission.STATUS_BAR,android.permission.SYSTEM_ALERT_WINDOW," \
           "android.permission.UPDATE_DEVICE_STATS,android.permission.USE_CREDENTIALS,android.permission.VIBRATE," \
           "android.permission.WAKE_LOCK,android.permission.WRITE_APN_SETTINGS,android.permission.WRITE_SETTINGS," \
           "android.permission.WRITE_SMS,android.permission.WRITE_EXTERNAL_STORAGE"
cols_featureList = [feature.strip() for feature in features.split(',')]
feature_size = 158
embedding_length = 10
embedding_dim = 768

# exp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 50

# reformerLayer
d_in = 384
reform_dim = 256
n_heads = 8
d_values = reform_dim // n_heads

# combinerLayer
combine_n_heads = 4
mark_feature = 64
combine_d_values = mark_feature // combine_n_heads

# featureDetect
cure_query_num = 4
cureFeatureNum = 8

# prompt
Example = "标题： 安卓应用安全风险分析报告\n\
\n\
输入：关键特征： 描述被识别为潜在安全风险的权限和函数。\n\
关联特征： 对于每个关键特征，列出3-5个有重要关联的特征。\n\
\n\
应用名称： Name\n\
检测日期： Date\n\
关键特征1：Feature1\n\
关键特征1关联特征：Assistance1\n\
关键特征2：Feature2\n\
关键特征2关联特征：Assistance2\n\
关键特征3：Feature3\n\
关键特征3关联特征：Assistance3\n\
关键特征4：Feature4\n\
关键特征4关联特征：Assistance4\n\
指令：\n\
\n\
风险评估： 根据关键特征及其关联特征，分析潜在的安全风险类型（如隐私泄露、恶意行为等）。\n\
关键特征分析： 解释为什么这些特征被认为是危险的，并分析它们是如何相互关联的。\n\
风险模式： 根据分析，识别出常见的风险模式，说明这些模式对应用安全的具体影响。\n\
建议和预防措施： 提供针对识别出的风险模式的建议和预防措施。\n\
输出格式：\n\
\n\
摘要： 简要概述发现的主要风险及其潜在影响。\n\
详细分析： 详细报告每个关键特征及其关联特征的风险分析，包括风险类型、关键特征详细解释、关联特征详细解释、 关键、关联特征间的相互作用，以及风险模式的识别。\n\
建议： 基于分析，提出具体的建议和预防措施。"


