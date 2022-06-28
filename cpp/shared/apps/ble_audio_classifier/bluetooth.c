
#include <stdio.h>
#include "sl_bluetooth.h"
#include "gatt_db.h"
#include "ble_audio_classifier.h"



static void notify_keyword_detected(uint8_t class_id, uint8_t confidence);


// The advertising set handle allocated from Bluetooth stack.
static uint8_t advertising_set_handle = 0xff;


/**************************************************************************//**
 * Bluetooth stack event handler.
 * This overrides the dummy weak implementation.
 *
 * @param[in] evt Event coming from the Bluetooth stack.
 *****************************************************************************/
void sl_bt_on_event(sl_bt_msg_t *evt)
{
  sl_status_t sc;
  bd_addr address;
  uint8_t address_type;
  uint8_t system_id[8];

  switch (SL_BT_MSG_ID(evt->header)) {
    // -------------------------------
    // This event indicates the device has started and the radio is ready.
    // Do not call any stack command before receiving this boot event!
    case sl_bt_evt_system_boot_id:
      // Extract unique ID from BT Address.
      sc = sl_bt_system_get_identity_address(&address, &address_type);

      // Pad and reverse unique ID to get System ID.
      system_id[0] = address.addr[5];
      system_id[1] = address.addr[4];
      system_id[2] = address.addr[3];
      system_id[3] = 0xFF;
      system_id[4] = 0xFE;
      system_id[5] = address.addr[2];
      system_id[6] = address.addr[1];
      system_id[7] = address.addr[0];

      sc = sl_bt_gatt_server_write_attribute_value(gattdb_system_id,
                                                   0,
                                                   sizeof(system_id),
                                                   system_id);
      // Create an advertising set.
      sc = sl_bt_advertiser_create_set(&advertising_set_handle);

      // Generate data for advertising
      sc = sl_bt_legacy_advertiser_generate_data(advertising_set_handle,
                                                 sl_bt_advertiser_general_discoverable);

      // Set advertising interval to 100ms.
      sc = sl_bt_advertiser_set_timing(
        advertising_set_handle,
        160, // min. adv. interval (milliseconds * 1.6)
        160, // max. adv. interval (milliseconds * 1.6)
        0,   // adv. duration
        0);  // max. num. adv. events

      // Start advertising and enable connections.
      sc = sl_bt_legacy_advertiser_start(advertising_set_handle,
                                         sl_bt_advertiser_connectable_scannable);

      ble_audio_classifier_set_detection_callback(notify_keyword_detected);

    //   // Button events can be received from now on.
    //   sl_button_enable(SL_SIMPLE_BUTTON_INSTANCE(0));

    //   // Check the report button state, then update the characteristic and
    //   // send notification.
    //   sc = update_report_button_characteristic();
    //   app_log_status_error(sc);

    //   if (sc == SL_STATUS_OK) {
    //     sc = send_report_button_notification();
    //     app_log_status_error(sc);
    //   }
      break;

    // -------------------------------
    // This event indicates that a new connection was opened.
    case sl_bt_evt_connection_opened_id:
      printf("Connection opened\n");
      ble_audio_classifier_start();
      break;

    // -------------------------------
    // This event indicates that a connection was closed.
    case sl_bt_evt_connection_closed_id:
      printf("Connection closed\n");
      ble_audio_classifier_stop();

      // Generate data for advertising
      sc = sl_bt_legacy_advertiser_generate_data(advertising_set_handle,
                                                 sl_bt_advertiser_general_discoverable);

      // Restart advertising after client has disconnected.
      sc = sl_bt_legacy_advertiser_start(advertising_set_handle,
                                         sl_bt_advertiser_connectable_scannable);
      break;

    // -------------------------------
    // This event indicates that the value of an attribute in the local GATT
    // database was changed by a remote GATT client.
    case sl_bt_evt_gatt_server_attribute_value_id:
      // The value of the gattdb_led_control characteristic was changed.
      // if (gattdb_led_control == evt->data.evt_gatt_server_characteristic_status.characteristic) {
      //   uint8_t data_recv;
      //   size_t data_recv_len;

      //   // Read characteristic value.
      //   sc = sl_bt_gatt_server_read_attribute_value(gattdb_led_control,
      //                                               0,
      //                                               sizeof(data_recv),
      //                                               &data_recv_len,
      //                                               &data_recv);
      //   (void)data_recv_len;

      //   if (sc != SL_STATUS_OK) {
      //     break;
      //   }

      //   // Toggle LED.
      //   if (data_recv == 0x00) {
      //     //sl_led_turn_off(SL_SIMPLE_LED_INSTANCE(0));
      //     printf("LED off.\n");
      //   } else if (data_recv == 0x01) {
      //     //sl_led_turn_on(SL_SIMPLE_LED_INSTANCE(0));
      //     printf("LED on.\n");
      //   } else {
      //     printf("Invalid attribute value: 0x%02x\n", (int)data_recv);
      //   }
      // }
      break;

    // -------------------------------
    // This event occurs when the remote device enabled or disabled the
    // notification.
    case sl_bt_evt_gatt_server_characteristic_status_id:
      // if (gattdb_report_button == evt->data.evt_gatt_server_characteristic_status.characteristic) {
      //   // A local Client Characteristic Configuration descriptor was changed in
      //   // the gattdb_report_button characteristic.
      //   if (evt->data.evt_gatt_server_characteristic_status.client_config_flags
      //       & sl_bt_gatt_notification) {
      //     // The client just enabled the notification. Send notification of the
      //     // current button state stored in the local GATT table.
      //     printf("Notification enabled\n");

      //    // sc = send_report_button_notification();
      //   } else {
      //     printf("Notification disabled\n");
      //   }
      // }
      break;

    ///////////////////////////////////////////////////////////////////////////
    // Add additional event handlers here as your application requires!      //
    ///////////////////////////////////////////////////////////////////////////

    // -------------------------------
    // Default event handler.
    default:
      break;
  }
}


static void notify_keyword_detected(uint8_t class_id, uint8_t confidence)
{
  sl_status_t sc;
  uint8_t data[32];

  int len = sprintf((char*)data, "%u,%u", class_id, confidence);

  sc = sl_bt_gatt_server_write_attribute_value(gattdb_command,
                                               0,
                                               len,
                                               data);
  if (sc != SL_STATUS_OK) {
    printf("Failed to write GATT DB, err:%d\n", sc);
    return;
  }

  sc = sl_bt_gatt_server_notify_all(gattdb_command,
                                    len,
                                    data);
  if (sc != SL_STATUS_OK) {
    printf("Failed to send notification, err:%d\n", sc);
    return;
  }
}